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
        self.delta_cache = {}    # mapping: block_idx -> list of computed delta tensors
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
        """
        For a single state:
          - If the cache list is a consecutive range (e.g. 10 to 21), then:
              * On a compute step:
                  - For blocks in that range: if block_idx equals the last block
                    (e.g. 21), compute the contiguous delta as
                        output(21) - output(9)
                    (where 9 is assumed to be the output of block 10-1).
                  - For other blocks in the range, compute normally.
              * On a cache step:
                  - For any block in that range, if the contiguous delta is available,
                    simply add the cached delta.
          - Otherwise, use the usual per-block caching logic.
        """
        sample_optim_flag = True
        # Before step_start: compute normally.
        if sample_optim_flag:
            if step_idx >= self.step_start:
                if step_idx in [6, 18, 26, 32, 38, 42, 46, 48]:
                    
                    
                    if block_idx in [0,2,23,31,33]: #[0, 2, [4, 7], [10,21], 23, [25, 27], 31, 33]
                        prev_hidden_states = hidden_states.clone()
                        hidden_states = block(hidden_states, *args, **kwargs)
                        delta = hidden_states - prev_hidden_states
                        self.delta_cache[block_idx] = delta
                        return hidden_states
                    elif block_idx in [3,9,24]:
                        
                        hidden_states = block(hidden_states, *args, **kwargs)
                        self.prev_hidden = hidden_states.clone()
                        return hidden_states
                    
                    elif block_idx in [7,21,27]:
                        hidden_states = block(hidden_states, *args, **kwargs)
                        cache = hidden_states - self.prev_hidden
                        self.delta_cache[block_idx] = cache
                        return hidden_states
                    else:
                        hidden_states = block(hidden_states, *args, **kwargs)
                        return hidden_states
                else:
                    if block_idx in [1, 3, 8, 9, 22, 24, 28, 29, 30, 32, 34, 35, 36, 37, 38, 39]: 
                        hidden_states = block(hidden_states, *args, **kwargs)
                        return hidden_states
                    elif block_idx in [0,2,23,31,33]:
                        hidden_states += self.delta_cache.get(block_idx)
                        return hidden_states
                    
                    elif block_idx in [7,21,27]:
                        hidden_states += self.delta_cache.get(block_idx)
                        return hidden_states
                    
                    elif block_idx in [4,5,6,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,25, 26, 27]:
                        return hidden_states
                    else:
                        hidden_states = block(hidden_states, *args, **kwargs)
                        return hidden_states
                        # print(len(self.uncontinguous_cache))
        
                    
            elif step_idx < self.step_start:
                hidden_states = block(hidden_states, *args, **kwargs)
                return hidden_states
        else:    
            if step_idx >= self.step_start:
                if (step_idx - self.step_start) % self.step_interval == 0:
                    
                    
                    if block_idx in [0,2,23,31,33]: #[0, 2, [4, 7], [10,21], 23, [25, 27], 31, 33]
                        prev_hidden_states = hidden_states.clone()
                        hidden_states = block(hidden_states, *args, **kwargs)
                        delta = hidden_states - prev_hidden_states
                        self.delta_cache[block_idx] = delta
                        return hidden_states
                    elif block_idx in [3,9,24]:
                        
                        hidden_states = block(hidden_states, *args, **kwargs)
                        self.prev_hidden = hidden_states.clone()
                        return hidden_states
                    
                    elif block_idx in [7,21,27]:
                        hidden_states = block(hidden_states, *args, **kwargs)
                        cache = hidden_states - self.prev_hidden
                        self.delta_cache[block_idx] = cache
                        return hidden_states
                    else:
                        hidden_states = block(hidden_states, *args, **kwargs)
                        return hidden_states
                else:
                    if block_idx in [1, 3, 8, 9, 22, 24, 28, 29, 30, 32, 34, 35, 36, 37, 38, 39]: 
                        hidden_states = block(hidden_states, *args, **kwargs)
                        return hidden_states
                    elif block_idx in [0,2,23,31,33]:
                        hidden_states += self.delta_cache.get(block_idx)
                        return hidden_states
                    
                    elif block_idx in [7,21,27]:
                        hidden_states += self.delta_cache.get(block_idx)
                        return hidden_states
                    
                    elif block_idx in [4,5,6,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,25, 26, 27]:
                        return hidden_states
                    else:
                        hidden_states = block(hidden_states, *args, **kwargs)
                        return hidden_states
                        # print(len(self.uncontinguous_cache))
        
                    
            elif step_idx < self.step_start:
                hidden_states = block(hidden_states, *args, **kwargs)
                return hidden_states
        

    def forward_double(self, block, time_step, block_idx, hidden_states, condition_state,*args, **kwargs
        ):  
            """
            Args:
                block: The block in the DiT module.
                time_step: The current time step.
                block_idx: The index of the block.
                hidden_states: The hidden states.
                *args: Additional arguments.
                **kwargs: Additional keyword arguments.
            """
            if not self.use_cache_double(time_step, block_idx):
                old_hidden_states = hidden_states
                old_condition_states = condition_state
                if isinstance(block, list):
                    for blk in block:
                        hidden_states, condition_state = blk(hidden_states, condition_state, *args, **kwargs)
                else:
                    hidden_states, condition_state = block(hidden_states, condition_state, *args, **kwargs)
                self.update_cache_double(hidden_states, old_hidden_states,
                                        condition_state, old_condition_states, time_step, block_idx)
            else:
                delta_double = self.get_cache_double(time_step, block_idx)
                hidden_states += delta_double[0]
                condition_state += delta_double[1]
            return hidden_states, condition_state
    # add cache double
    def use_cache_double(self, time_step, block_idx):
        if time_step < self.step_start:
            return False
        else:
            diftime = time_step - self.step_start
            if diftime not in self.time_cache:
                self.time_cache[diftime] = diftime % self.step_interval == 0
            if self.time_cache[diftime]:
                return False
            elif block_idx < self.block_start_double or block_idx > self.block_end_double:
                return False
            else:
                return True


    def use_cache(self, time_step, block_idx):
        if time_step < self.step_start:
            return False
        else:
            diftime = time_step - self.step_start
            if diftime not in self.time_cache:
                self.time_cache[diftime] = diftime % self.step_interval == 0
            if self.time_cache[diftime]:
                return False
            elif block_idx < self.block_start or block_idx > self.block_end:
                return False
            else:
                return True

    def get_cache(self, time_step, block_idx):
        if block_idx == self.block_start:
            return self.cache
        else:
            return 0
    
    def get_cache_double(self, time_step, block_idx):
        if block_idx == self.block_start_double:
            return self.cache_double
        else:
            return (0,0)

    def update_cache(self, hidden_states, old_hidden_states, time_step, block_idx):
        diftime = time_step - self.step_start
        # when (time_step - self.step_start) % self.step_interval == 0:
        if time_step >= self.step_start and self.time_cache[diftime]:
            if block_idx == self.block_start:
                self.cache = old_hidden_states.clone()
            elif block_idx == self.block_end:
                self.cache = hidden_states - self.cache

    def update_cache_double(self, hidden_states, old_hidden_states, condition_state, old_condition_states, time_step, block_idx):
        diftime = time_step - self.step_start
        # when (time_step - self.step_start) % self.step_interval == 0:
        if time_step >= self.step_start and self.time_cache[diftime]:
            if block_idx == self.block_start_double:
                self.cache_double = (old_hidden_states.clone(), old_condition_states.clone())
            elif block_idx == self.block_end_double:
                self.cache_double = (hidden_states - self.cache_double[0], condition_state - self.cache_double[1])
