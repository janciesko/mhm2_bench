#include "upcxx_utils/reduce_prefix.hpp"

using upcxx::future;

namespace upcxx_utils {

in_order_binary_tree_node::in_order_binary_tree_node(intrank_t _me, intrank_t _n, intrank_t skip_regular_local_ranks) {
  reset(_me, _n);
  init(skip_regular_local_ranks);
}

bool in_order_binary_tree_node::leftmost() const { return me == root || (me < root && parent > me && me - my_stride < 0); }

bool in_order_binary_tree_node::rightmost() const {
  intrank_t next_stride = me + my_stride;
  bool ret = me == root || (me > root && parent < me);
  if (!ret) {
    return false;
  } else if (ret && next_stride >= root * 2) {
    // full tree at this level
    return true;
  } else {
    // possibly but recursively determine if all parents are rightmost too
    return in_order_binary_tree_node(parent, n).rightmost();
  }
}

void in_order_binary_tree_node::reset(intrank_t _me, intrank_t _n) {
  me = _me;
  n = _n;
  parent = n;
  left = n;
  right = n;
  my_level = -1;
  my_stride = 0;
  height = 0;
  root = 0;
}

void in_order_binary_tree_node::init(intrank_t skip_regular_local_ranks) {
  if (skip_regular_local_ranks == 0) {
    init();
  } else {
    assert(n % skip_regular_local_ranks == 0);
    if (me % skip_regular_local_ranks == 0) {
      // first scale down the ranks
      reset(me / skip_regular_local_ranks, n / skip_regular_local_ranks);
      DBG_VERBOSE("Participating - skip_regular_local_ranks=", skip_regular_local_ranks, ", effective me=", me, " n=", n, "\n");
      init();
      // then scale all values to original size
      me *= skip_regular_local_ranks;
      n *= skip_regular_local_ranks;
      parent *= skip_regular_local_ranks;
      left *= skip_regular_local_ranks;
      right *= skip_regular_local_ranks;
      my_stride *= skip_regular_local_ranks;
      root *= skip_regular_local_ranks;
    } else {
      DBG_VERBOSE("Not participating - skip_regular_local_ranks=", skip_regular_local_ranks, "\n");
      me = n;  // does not participate
      assert(left == n);
      assert(right == n);
      assert(parent == n);
    }
  }
}

void in_order_binary_tree_node::init() {
  int parent_step = 1, level_stride = 2, child_step = 0, level = 0;

  height = 0;
  while (root < n) {
    height++;
    if (me % level_stride == root) {
      assert(my_level == -1);  // only one level matches a given rank
      my_level = level;
      my_stride = level_stride;
      intrank_t my_child_step = child_step;
      while (my_child_step) {
        // DBG_VERBOSE("my_child_step=", my_child_step, " left=", left, " right=", right, " level=", level, "\n");
        // if there is a lower level the left child is the very next level
        if (left == n) {
          left = me - my_child_step;
        }
        if (me + my_child_step < n) {
          right = me + my_child_step;
          break;
        }
        my_child_step >>= 1;
      }
    }
    if (my_level >= 0 && my_level + 1 == level) {
      // test which rank is my parent is at this level or to the left at a higher level
      int left_parent = me - child_step;
      int right_parent = me + child_step;
      if (right_parent < n && right_parent % level_stride == root) {
        parent = right_parent;
      } else if (left_parent > 0) {
        parent = left_parent;
      }
    }
    level++;
    if (root + parent_step < n) {
      root += parent_step;
    } else {
      break;
    }
    child_step = parent_step;
    level_stride <<= 1;
    parent_step <<= 1;
  }
  DBG_VERBOSE("me=", me, " n=", n, " parent=", parent, " left=", left, " right=", right, " root=", root, " my_level=", my_level,
              " my_stride=", my_stride, " height=", height, "\n");
  assert(root < n);
  assert(parent != n || root == me);
  assert(left == n || left < me);
  assert(right == n || me < right);
}

future<> binary_tree_steps::get_future() const {
  return when_all(dst_is_partial_left_me.get_future(), scratch_is_partial_right.get_future(),
                  scratch_is_partial_to_parent.get_future(), sent_partial_to_parent.get_future(),
                  scratch_is_partial_from_parent.get_future(), sent_left_child.get_future(), sent_right_child.get_future(),
                  dst_is_ready.get_future());
}
// up phase is done

bool binary_tree_steps::up_ready() const {
  return dst_is_partial_left_me.get_future().is_ready() && scratch_is_partial_right.get_future().is_ready() &&
         scratch_is_partial_to_parent.get_future().is_ready() && sent_partial_to_parent.get_future().is_ready();
}

future<> binary_tree_steps::get_up_future() const {
  return when_all(dst_is_partial_left_me.get_future(), scratch_is_partial_right.get_future(),
                  scratch_is_partial_to_parent.get_future(), sent_partial_to_parent.get_future());
}
// down phase is done

bool binary_tree_steps::down_ready() const {
  return scratch_is_partial_from_parent.get_future().is_ready() && sent_left_child.get_future().is_ready() &&
         sent_right_child.get_future().is_ready();
}

future<> binary_tree_steps::get_down_future() const {
  return when_all(scratch_is_partial_from_parent.get_future(), sent_left_child.get_future(), sent_right_child.get_future());
}

// Reduce compile time by making templates instantiations of common types
// extern templates happen CMakeList and reduce_prefix-extern-template.in.cpp

/*
 * This is now handled by CMakeLists.txt
 *
    MACRO_REDUCE_PREFIX(float,    upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<true>  COMMA true>,  template);
    MACRO_REDUCE_PREFIX(float,    upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<false> COMMA true>,  template);
    MACRO_REDUCE_PREFIX(float,    upcxx::detail::op_wrap<upcxx::detail::opfn_add                COMMA true>,  template);

    MACRO_REDUCE_PREFIX(double,   upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<true>  COMMA true>,  template);
    MACRO_REDUCE_PREFIX(double,   upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<false> COMMA true>,  template);
    MACRO_REDUCE_PREFIX(double,   upcxx::detail::op_wrap<upcxx::detail::opfn_add                COMMA true>,  template);

    MACRO_REDUCE_PREFIX(int,      upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<true>  COMMA true>,  template);
    MACRO_REDUCE_PREFIX(int,      upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<false> COMMA true>,  template);
    MACRO_REDUCE_PREFIX(int,      upcxx::detail::op_wrap<upcxx::detail::opfn_add                COMMA true>,  template);

    MACRO_REDUCE_PREFIX(uint64_t, upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<true>  COMMA true>,  template);
    MACRO_REDUCE_PREFIX(uint64_t, upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<false> COMMA true>,  template);
    MACRO_REDUCE_PREFIX(uint64_t, upcxx::detail::op_wrap<upcxx::detail::opfn_add                COMMA true>,  template);

    MACRO_REDUCE_PREFIX(int64_t,  upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<true>  COMMA true>,  template);
    MACRO_REDUCE_PREFIX(int64_t,  upcxx::detail::op_wrap<upcxx::detail::opfn_min_not_max<false> COMMA true>,  template);
    MACRO_REDUCE_PREFIX(int64_t,  upcxx::detail::op_wrap<upcxx::detail::opfn_add                COMMA true>,  template);
*/
};  // namespace upcxx_utils
