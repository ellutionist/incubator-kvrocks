#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "rocksdb/db.h"
#include "util.h"

template <typename T>
class BlockingQueue {
  std::condition_variable _cvCanPop;
  std::mutex _sync;
  std::queue<T> _qu;
  bool _bShutdown = false;

 public:
  void Push(const T& item) {
    {
      std::unique_lock<std::mutex> lock(_sync);
      _qu.push(item);
    }
    _cvCanPop.notify_one();
  }

  void Push(T&& rhs) {
    {
      std::unique_lock<std::mutex> lock(_sync);
      _qu.push(std::move(rhs));
    }
    _cvCanPop.notify_one();
  }

  void RequestShutdown() {
    {
      std::unique_lock<std::mutex> lock(_sync);
      _bShutdown = true;
    }
    _cvCanPop.notify_all();
  }

  bool Closed() { return _bShutdown; }

  bool Pop(T& item, int seconds = -1) {
    std::unique_lock<std::mutex> lock(_sync);
    for (;;) {
      if (_qu.empty()) {
        if (_bShutdown) {
          return false;
        }
      } else {
        break;
      }

      if (seconds > 0) {
        _cvCanPop.wait_for(lock, std::chrono::seconds(seconds));
      } else {
        _cvCanPop.wait(lock);
      }
    }
    item = std::move(_qu.front());
    _qu.pop();
    return true;
  }
};

struct KvPair {
  std::string key;
  std::string value;
  std::optional<int> ttl;
};

struct PikaValue {
  std::string_view value;
  uint32_t expire;

  static uint32_t LittleEndian(const char* ptr) {
    return static_cast<uint32_t>(static_cast<uint8_t>(ptr[0])) |
           (static_cast<uint32_t>(static_cast<uint8_t>(ptr[1])) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(ptr[2])) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(ptr[3])) << 24);
  }

  explicit PikaValue(std::string_view raw)
      : value(raw.substr(0, raw.size() - 4)), expire(LittleEndian(raw.data() + (raw.size() - 4))) {}

  std::optional<int> Ttl() {
    if (expire == 0) {
      return std::nullopt;
    }

    auto now = Util::GetTimeStamp<std::chrono::seconds>();

    return int(expire - now);
  }

  bool Expired() {
    auto ttl = Ttl();
    return (ttl.has_value() && ttl <= 0);
  }
};

class PikaDB {
 public:
  using db_type = std::unique_ptr<rocksdb::DB>;
  using queue_type = BlockingQueue<KvPair>;

 public:
  explicit PikaDB(std::string db_path, std::vector<std::string>&& prefixes, int n_scan_threads = 1)
      : db_path_(std::move(db_path)), prefixes_(std::move(prefixes)), n_threads_(n_scan_threads) {
    if (prefixes_.empty()) {
      prefixes_.emplace_back("");
    }
  }

  PikaDB(const PikaDB&) = delete;
  PikaDB(PikaDB&&) = delete;

  PikaDB&& operator=(const PikaDB&) = delete;
  PikaDB&& operator=(const PikaDB&&) = delete;

  ~PikaDB() {
    if (db_) {
      db_->Close();
    }
  }

 private:
  std::string db_path_;
  db_type db_;
  std::mutex prefix_mutex_;
  std::vector<std::string> prefixes_;

  queue_type queue_;

  int n_threads_ = 1;

  std::vector<std::thread> threads_;

 private:
  void join() {
    for (auto&& t : threads_) t.join();
  }

  std::optional<std::string> nextPrefix() {
    std::lock_guard<std::mutex> g{prefix_mutex_};

    if (prefixes_.empty()) {
      return std::nullopt;
    }

    using ret_type = decltype(nextPrefix());
    ret_type ret = {std::move(prefixes_.back())};

    prefixes_.pop_back();
    return ret;
  }

  int doScan(const std::string& prefix) {
    int count = 0;
    rocksdb::ReadOptions opts;
    auto iter = std::unique_ptr<rocksdb::Iterator>(db_->NewIterator(opts));

    for (iter->Seek(prefix); iter->Valid() && iter->key().starts_with(prefix); iter->Next()) {
      PikaValue pv{iter->value().ToStringView()};
      auto ttl = pv.Ttl();
      if (ttl.has_value() && ttl <= 0) {
        continue;
      }

      KvPair pair;
      pair.key = iter->key().ToStringView();
      pair.value = pv.value;
      pair.ttl = ttl;

      queue_.Push(std::move(pair));
      count++;
    }

    LOG(INFO) << count << " keys for prefix \"" << prefix << "\"";
    return count;
  }

  void threadMain(int idx) {
    LOG(INFO) << "scan thread " << idx << " start.";
    int count = 0;
    while (true) {
      auto prefix = nextPrefix();
      if (!prefix.has_value()) {
        break;
      }

      count += doScan(prefix.value());
    }
    LOG(INFO) << "scan thread " << idx << " stop with " << count << " keys.";
  }

 public:
  queue_type& GetQueue() { return queue_; }

  void Start() {
    rocksdb::DB* db;
    rocksdb::Options opts;
    opts.create_if_missing = false;

    auto s = rocksdb::DB::OpenForReadOnly(opts, db_path_, &db);
    if (!s.ok()) {
      LOG(ERROR) << s.ToString();
    }

    assert(s.ok());

    db_.reset(db);

    for (int i = 0; i < n_threads_; i++) {
      threads_.emplace_back([this, i] { threadMain(i); });
    }
  }

  void Join() { join(); }
};
