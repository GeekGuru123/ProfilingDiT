class DitCache:
    def __init__(
        self, 
        step_start, 
        step_interval,  # should be set to 4
        block_start, 
        num_blocks, 
        block_start_double=None,
        num_blocks_double=None, 
        block_cache_list=None, 
        **kwargs
    ):
        self.step_start = step_start
        self.step_interval = step_interval  # e.g., 4
        self.block_start = block_start
        self.num_blocks = num_blocks
        self.block_end = block_start + num_blocks - 1

        # Double blocks (if needed)
        self.block_start_double = block_start_double
        self.num_blocks_double = num_blocks_double
        self.block_end_double = (
            block_start_double + num_blocks_double - 1 
            if block_start_double is not None 
            else None
        )
        
        # Original caches (unused in the new scheme)
        self.cache = True
        self.cache_double = None
        self.time_cache = {}  # not used now
        self.pre_block_idx = 0
        # Provided list of blocks to use cached delta
        self.block_cache_list = block_cache_list
        # if block_cache_list is not None and len(block_cache_list) > 0:
        #     sorted_cache = sorted(block_cache_list)
        #     # Check if the provided cache list is consecutive.
        #     self.contiguous_cache = all(sorted_cache[i+1] - sorted_cache[i] == 1 for i in range(len(sorted_cache)-1))
        #     if self.contiguous_cache:
        #         self.contiguous_cache_start = sorted_cache[0]
        #         self.contiguous_cache_end = sorted_cache[-1]
        #         # These will hold the delta computed once for the entire contiguous range.
        #         self.contiguous_delta = None         # for forward_single
        #         self.contiguous_delta_double = None  # for forward_double
        #     else:
        #         self.contiguous_cache = False
        # else:
        #     self.contiguous_cache = False

        # For blocks not in the cache list, we compute normally and update per-block delta.
        # nums = [0, 2, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 27, 31, 33]

        # 结果存储
        self.continuous_ranges = []  # 连续数字的首尾
        self.non_continuous = []  # 没有连续的数字
        nums = self.block_cache_list
        # 先找连续的数字范围
        start = nums[0]
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1] + 1:  # 断开了
                if start != nums[i - 1]:  # 说明之前是一段连续的
                    self.continuous_ranges.append([start, nums[i - 1]])
                else:  # 单独的数字
                    self.non_continuous.append(start)
                start = nums[i]  # 重新开始

        # 处理最后一段
        if start != nums[-1]:
            self.continuous_ranges.append([start, nums[-1]])
        else:
            self.non_continuous.append(start)
        self.start_minus_1 = [start - 1 for start, _ in self.continuous_ranges]
        self.end_values = [end for _, end in self.continuous_ranges]
        print("连续数字的首尾：", self.continuous_ranges)
        print("没有连续的数字：", self.non_continuous)

        all_blocks = set(range(40))
        cached_blocks = set(self.block_cache_list) if self.block_cache_list is not None else set()
        self.not_cached_list = sorted(list(all_blocks - cached_blocks))

        # For each block idx in not_cached_list, keep the previous output and delta history.
        self.prev_hidden = None     # for forward_single
        self.delta_cache = []    # mapping: block_idx -> list of computed delta tensors
        self.continguous_cache = []
        self.uncontinguous_cache = []
        # Similarly for blocks that output two states.
        self.prev_hidden_double = {}    # mapping: block_idx -> (prev_hidden, prev_condition)
        self.delta_cache_double = {}    # mapping: block_idx -> list of (delta_hidden, delta_condition)

    def _is_compute_step(self, step_idx):
        """
        Returns True if we are in a compute step.
        A compute step occurs every `step_interval` steps (e.g., every 4 steps),
        starting at step_start. Before step_start, always compute normally.
        """
        if step_idx < self.step_start:
            return True
        return ((step_idx - self.step_start) % self.step_interval) == 0

    def forward_single(self, block, step_idx, block_idx, hidden_states, *args, **kwargs):

        # Before step_start: compute normally.
    
        
        if (step_idx - self.step_start) % self.step_interval == 0 and step_idx > self.step_start:
            hidden_states = block(hidden_states, *args, **kwargs)
            
            if block_idx in self.block_cache_list : #[0, 2, [4, 7], [10,21], 23, [25, 27], 31, 33]
                if block_idx in self.non_continuous:
                    temp = hidden_states.clone()
                    hidden_states = block(hidden_states, *args, **kwargs)
                    delta = hidden_states - temp
                    self.uncontinguous_cache.append(delta)
                    # print(len(self.uncontinguous_cache))
                    return hidden_states
                elif any(start <= block_idx <= end for start, end in self.continuous_ranges):
                    if any(block_idx == start for start, end in self.continuous_ranges):
                        hidden_states = block(hidden_states, *args, **kwargs)
                        self.prev_hidden = hidden_states.clone()
                        return hidden_states
                    elif any(block_idx == end for start, end in self.continuous_ranges):
                        hidden_states = block(hidden_states, *args, **kwargs)
                        self.continguous_cache.append(hidden_states - self.prev_hidden)
                        return hidden_states
                    elif any(start < block_idx < end for start, end in self.continuous_ranges):
                        hidden_states = block(hidden_states, *args, **kwargs)
                        return hidden_states
            else:
                hidden_states = block(hidden_states, *args, **kwargs)
                if block_idx in self.start_minus_1:
                    self.prev_hidden = hidden_states.clone()
                return hidden_states
        elif step_idx < self.step_start:
            hidden_states = block(hidden_states, *args, **kwargs)
            return hidden_states
        else:
            if block_idx in self.non_continuous: #[1, 3, 8, 9, 22, 24, 28, 29, 30, 32, 34, 35, 36, 37, 38, 39]
                # print(len(self.uncontinguous_cache))
                # print(self.uncontinguous_cache[0])
                # print(self.count1)
                index = self.non_continuous.index(block_idx)
                print(f'non {index}')
                print(len(self.uncontinguous_cache))
                hidden_states += self.uncontinguous_cache[index]
             
                return hidden_states
                
            elif any(start <= block_idx <= end for start, end in self.continuous_ranges):
                if any(block_idx == start for start, end in self.continuous_ranges):
                    hidden_states = block(hidden_states, *args, **kwargs)
                    return hidden_states
                elif any(block_idx == end for start, end in self.continuous_ranges):
                    index = self.end_values.index(block_idx)
                    print(f'is con {index}')
                    hidden_states += self.continguous_cache[index]
                    
                    return hidden_states
                elif any(start < block_idx < end for start, end in self.continuous_ranges):
                    return hidden_states

            else:
                hidden_states = block(hidden_states, *args, **kwargs)
                if (step_idx - self.step_start) % self.step_interval == self.step_interval - 1 and block_idx == 39:
                    self.continguous_cache = []
                    self.uncontinguous_cache = []
                 
                return hidden_states
            
        

    def forward_double(self, block, step_idx, block_idx, hidden_states, condition_state, *args, **kwargs):
        """
        For blocks outputting two states:
          - If the cache list is consecutive, then for block indices in that range:
              * On a compute step:
                  - For the final block in the contiguous range, compute the overall delta:
                        (output_hidden, output_condition) - (prev_hidden, prev_condition)
                    using the outputs from block (contiguous_cache_start - 1).
                  - For other blocks, compute normally.
              * On a cache step:
                  - For any block in that range, if the contiguous delta is available,
                    simply add it to the inputs.
          - Otherwise, use the usual per-block caching logic.
        """
        if step_idx < self.step_start:
            print(f"Step {step_idx} Block {block_idx} (double): Before step_start, computing normally.")
            if isinstance(block, list):
                for blk in block:
                    hidden_states, condition_state = blk(hidden_states, condition_state, *args, **kwargs)
            else:
                hidden_states, condition_state = block(hidden_states, condition_state, *args, **kwargs)
            return hidden_states, condition_state

        compute_step = self._is_compute_step(step_idx)
        
        # --- Check for contiguous caching in double case ---
        if self.contiguous_cache and (self.contiguous_cache_start <= block_idx <= self.contiguous_cache_end):
            if compute_step:
                if block_idx == self.contiguous_cache_end:
                    prev_hidden, prev_condition = self.prev_hidden_double.get(
                        self.contiguous_cache_start - 1, (hidden_states, condition_state)
                    )
                    if isinstance(block, list):
                        for blk in block:
                            hidden_states, condition_state = blk(hidden_states, condition_state, *args, **kwargs)
                    else:
                        hidden_states, condition_state = block(hidden_states, condition_state, *args, **kwargs)
                    delta_hidden = hidden_states - prev_hidden
                    delta_condition = condition_state - prev_condition
                    self.contiguous_delta_double = (delta_hidden, delta_condition)
                    # print(f"Step {step_idx} Block {block_idx} (contiguous, double): Compute step, computed contiguous delta.")
                else:
                    # print(f"Step {step_idx} Block {block_idx} (contiguous, double): Compute step, not final block, computing normally.")
                    if isinstance(block, list):
                        for blk in block:
                            hidden_states, condition_state = blk(hidden_states, condition_state, *args, **kwargs)
                    else:
                        hidden_states, condition_state = block(hidden_states, condition_state, *args, **kwargs)
                return hidden_states, condition_state
            else:
                if self.contiguous_delta_double is not None:
                    # print(f"Step {step_idx} Block {block_idx} (contiguous, double): Cache step, using contiguous cached delta.")
                    delta_hidden, delta_condition = self.contiguous_delta_double
                    hidden_states = hidden_states + delta_hidden
                    condition_state = condition_state + delta_condition
                    return hidden_states, condition_state
                else:
                    # print(f"Step {step_idx} Block {block_idx} (contiguous, double): Cache step, contiguous delta not available, computing normally.")
                    if isinstance(block, list):
                        for blk in block:
                            hidden_states, condition_state = blk(hidden_states, condition_state, *args, **kwargs)
                    else:
                        hidden_states, condition_state = block(hidden_states, condition_state, *args, **kwargs)
                    return hidden_states, condition_state

        # --- Regular per-block caching logic for double state ---
        if compute_step:
            if block_idx in self.not_cached_list:
                # print(f"Step {step_idx} Block {block_idx} (double): Compute step, computing delta.")
                prev_hidden, prev_condition = self.prev_hidden_double.get(
                    block_idx, (hidden_states, condition_state)
                )
                if isinstance(block, list):
                    for blk in block:
                        hidden_states, condition_state = blk(hidden_states, condition_state, *args, **kwargs)
                else:
                    hidden_states, condition_state = block(hidden_states, condition_state, *args, **kwargs)
                delta_hidden = hidden_states - prev_hidden
                delta_condition = condition_state - prev_condition
                if block_idx not in self.delta_cache_double:
                    self.delta_cache_double[block_idx] = []
                self.delta_cache_double[block_idx].append((delta_hidden, delta_condition))
                self.prev_hidden_double[block_idx] = (hidden_states.clone(), condition_state.clone())
                return hidden_states, condition_state
            else:
                # print(f"Step {step_idx} Block {block_idx} (double): Compute step, no caching applied.")
                if isinstance(block, list):
                    for blk in block:
                        hidden_states, condition_state = blk(hidden_states, condition_state, *args, **kwargs)
                else:
                    hidden_states, condition_state = block(hidden_states, condition_state, *args, **kwargs)
                return hidden_states, condition_state
        else:
            if (
                self.block_cache_list is not None and 
                block_idx in self.block_cache_list and 
                block_idx in self.delta_cache_double and 
                self.delta_cache_double[block_idx]
            ):
                # print(f"Step {step_idx} Block {block_idx} (double): Cache step, using cached delta.")
                delta_hidden, delta_condition = self.delta_cache_double[block_idx][-1]
                hidden_states = hidden_states + delta_hidden
                condition_state = condition_state + delta_condition
                return hidden_states, condition_state
            else:
                # print(f"Step {step_idx} Block {block_idx} (double): Cache step, no cached delta available, computing normally.")
                if isinstance(block, list):
                    for blk in block:
                        hidden_states, condition_state = blk(hidden_states, condition_state, *args, **kwargs)
                else:
                    hidden_states, condition_state = block(hidden_states, condition_state, *args, **kwargs)
                return hidden_states, condition_state
