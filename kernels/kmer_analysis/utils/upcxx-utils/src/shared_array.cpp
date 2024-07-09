#include "upcxx_utils/shared_array.hpp"

#include <atomic>
#include <upcxx/upcxx.hpp>

void upcxx_utils::AtomicOffsetSizeCapacity::set_capacity(size_t capacity, size_t start_offset) {
  // first ensure all values are 0;
  _capacity = 0;
  reset(0);
  if (start_offset > 0) {
    // set the size and then offset
    _size.store(start_offset);
    _offset.store(start_offset);
  }
  // last set the new capacity
  _capacity = capacity;
}

upcxx_utils::AtomicOffsetSizeCapacity::AtomicOffsetSizeCapacity(size_t capacity, size_t start_offset)
    : _capacity(0)
    , _size(0)
    , _offset(0) {
  // set the capacity last to avoid races
  set_capacity(capacity, start_offset);
}

size_t upcxx_utils::AtomicOffsetSizeCapacity::reset(size_t new_offset) {
  assert(_capacity >= new_offset);

  // first ensure no more appends can happen
  size_t oldOffset = _offset.fetch_add(_capacity);

  // next block until _size == oldOffset
  size_t oldSize, targetSize = oldOffset < _capacity ? oldOffset : _capacity;
  while ((oldSize = _size.load()) != targetSize) upcxx::progress();

  if (oldOffset > 0 && oldOffset < _capacity) {
    // careful here this may be a race condition if oldSize != oldOffset
    if (oldOffset != oldSize) throw runtime_error("reset() called when not full and size != offset! Race condition");
  } else if (oldOffset >= _capacity) {
    // careful here if size != capacity
    if (oldSize != _capacity) throw runtime_error("reset() called when full and size != capacity! Race condition");
  }
  // okay set size then offset
  size_t oldSize2 = _size.exchange(new_offset);
  if (oldSize2 != oldSize) throw runtime_error("size changed while reset() is executing");
  // finally set offset and allow new appends
  size_t oldOffset2 = _offset.exchange(new_offset);
  if (oldOffset2 < _capacity || oldOffset2 < oldOffset + _capacity)
    throw runtime_error("offset changed while reset() is executing");
  return oldSize;
}

bool upcxx_utils::AtomicOffsetSizeCapacity::ready() const {
  if (_capacity == 0) return false;
  // load size first, then possibly offset
  size_t size = _size.load();
  if (size > _capacity) return false;  // detected an undefined state
  if (size == _capacity) {
    assert(_offset.load() >= _capacity);
    return true;
  }
  size_t offset = _offset.load();
  if (offset < _capacity) {
    // not at capacity, so only ready if size and offset are equal
    return offset == size;
  }
  return false;
}

bool upcxx_utils::AtomicOffsetSizeCapacity::full() const {
  // not full if not initialized
  if (_capacity == 0) return false;
  // note, may not be is_ready()
  return _offset.load() >= _capacity;
}

bool upcxx_utils::AtomicOffsetSizeCapacity::empty() const {
  // not empty if not initialized!
  if (_capacity == 0) return false;
  return _offset.load() == 0;
}

size_t upcxx_utils::AtomicOffsetSizeCapacity::append(size_t len, CallbackFunc &callback) {
  DBG_VERBOSE("(len=", len, " callback=", &callback, ")\n");
  assert(len > 0);
  // step 1 do nothing if full or not initialized
  if (_capacity == 0) {
    DBG_VERBOSE("capacity==0, ret=0\n");
    return 0;
  }
  size_t old_offset;
  // step 2 increment _offset, allow overflow of capacity
  old_offset = _offset.fetch_add(len);
  if (old_offset < _capacity) {
    // will append at least some of data
    size_t max_len = _capacity - old_offset;
    size_t add_len = len;
    if (max_len < len) {
      // will fill to capacity
      // (leave excess length in _offset)
      assert(_offset.load() >= _capacity);
      add_len = max_len;
      assert(add_len > 0);
    } else {
      // add the entire len
    }
    assert(old_offset + add_len <= _capacity);

    // step 2 perform the callback with the allocated offset and length
    callback(old_offset, add_len);

    // step 3 add to size, signal readiness
    size_t old_size = _size.fetch_add(add_len);
    assert(old_size + add_len <= _capacity);
    if (old_offset + add_len >= _capacity) {
      assert(old_offset + add_len == _capacity);
      DBG_VERBOSE("appended with no room left, old_size=", old_size, " old_offset=", old_offset, " ret=", add_len, "\n");
      return add_len;
    } else {
      assert(add_len == len);
      DBG_VERBOSE("appended with room left (", (_capacity - old_offset - add_len), "). old_size=", old_size,
                  " old_offset=", old_offset, " ret=", len + 1, "\n");
      return len + 1;
    }
  } else {
    // no data will be appended
    // leave offset over capacity to avoid races
    DBG_VERBOSE("is now over full, ret=0\n");
    return 0;
  }
}
