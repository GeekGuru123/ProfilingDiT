class DitCache:
    def __init__(
        self, 
        step_start, 
        step_interval,  # should be set to 4
        block_start, 
        num_blocks, 
        block_start_double=None,
        num_blocks_double=None, 
        block_cache_list_background = None,
        block_cache_list_foreground = None,
        step_cache_list = None,
        **kwargs
    ):
        self.step_start = step_start
        self.step_interval = step_interval  # step interval T_s = f(s) setting, e.g., 4
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
        self.block_cache_list_background = block_cache_list_background
        self.block_cache_list_foreground = block_cache_list_foreground
        self.step_cache_list = step_cache_list
        self.uncontinuous = []  # Isolate number
        self.start = []  # Previous number of continuous range: start number of continuous range - 1
        self.end = []    # End number of continuous range
        self.continuous_numbers = []  # All continuous numbers
        self.missing_numbers = []  # Numbers remaining after removing block_cache_list from 0-39
        
        self.uncontinuous1 = [] # Isolate number
        self.start1 = []  # Previous number of continuous range: start number of continuous range - 1
        self.end1 = []    # End number of continuous range
        self.continuous_numbers1 = []  # All continuous numbers
        self.missing_numbers1 = []  # Numbers remaining after removing block_cache_list from 0-39

        n = len(self.block_cache_list_background)
        i = 0

        while i < n:
            # Record the starting number of the current continuous range
            current = self.block_cache_list_background[i]
            j = i
            # Find consecutive numbers
            while j + 1 < n and self.block_cache_list_background[j + 1] == self.block_cache_list_background[j] + 1:
                j += 1
            if i == j:  # Only one number, not continuous
                self.uncontinuous.append(self.block_cache_list_background[i])
            else:  # Discover continuous range
                self.start.append(self.block_cache_list_background[i] - 1)
                self.end.append(self.block_cache_list_background[j])
                # Record all continuous numbers
                self.continuous_numbers.extend(self.block_cache_list_background[i:j+1])
            i = j + 1

        # Calculate missing numbers (numbers not in block_cache_list in the range 0-39)
        full_range = set(range(40))  # Generate complete set of 0-39
        self.missing_numbers = sorted(full_range - set(self.block_cache_list_background))

        
        m = len(self.block_cache_list_foreground)
        i1 = 0

        while i1 < m:
            # Record the starting number of the current continuous range
            current = self.block_cache_list_foreground[i1]
            j1 = i1
            # Find consecutive numbers
            while j1 + 1 < m and self.block_cache_list_foreground[j1 + 1] == self.block_cache_list_foreground[j1] + 1:
                j1 += 1
            if i1 == j1:  # Only one number, not continuous
                self.uncontinuous1.append(self.block_cache_list_foreground[i1])
            else:  # Discover continuous range
                self.start1.append(self.block_cache_list_foreground[i1] - 1)
                self.end1.append(self.block_cache_list_foreground[j1])
                # Record all continuous numbers
                self.continuous_numbers1.extend(self.block_cache_list_foreground[i1:j1+1])
            i1 = j1 + 1

        # Calculate missing numbers (numbers not in block_cache_list in the range 0-39)
        full_range = set(range(40))  # Generate complete set of 0-39
        self.missing_numbers1 = sorted(full_range - set(self.block_cache_list_foreground))

        # For each block idx in not_cached_list, keep the previous output and delta history.
        self.prev_hidden = None     # used
        self.delta_cache = {}    # used

        # Similarly for blocks that output two states.
        self.prev_hidden_double = None    # mapping: block_idx -> (prev_hidden, prev_condition)
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

    def forward_single_original(self, block, step_idx, block_idx, hidden_states, *args, **kwargs):
        """
        Step-wise caching with faster forward and slower backward
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
        sample_optim_flag = False
        # Before step_start: compute normally.
        if sample_optim_flag: #前快后慢step wise
            if step_idx >= self.step_start:
                # \mathcal{S}_{\text{comp}} = \{s \mid s \bmod T_s = 0, s > s_0\}
                if step_idx in self.step_cache_list:    
                    # if step_idx in [6, 8, 12, 18, 26, 38]:
                    # h_s = h_s + \Delta_{\text{cache}}^{(i)}
                    if block_idx in self.uncontinuous1: #[0, 2, [4, 7], [10,21], 23, [25, 27], 31, 33]
                        prev_hidden_states = hidden_states.clone()
                        hidden_states = block(hidden_states, *args, **kwargs)
                        delta = hidden_states - prev_hidden_states
                        self.delta_cache[block_idx] = delta
                        return hidden_states
                    elif block_idx in self.start1:
                        
                        hidden_states = block(hidden_states, *args, **kwargs)  
                        self.prev_hidden = hidden_states.clone()
                        return hidden_states
                    
                    elif block_idx in self.end1:
                        hidden_states = block(hidden_states, *args, **kwargs)
                        cache = hidden_states - self.prev_hidden
                        self.delta_cache[block_idx] = cache
                        return hidden_states
                    else:
                        hidden_states = block(hidden_states, *args, **kwargs)
                        return hidden_states
                else:
                    if block_idx in self.missing_numbers1: 
                        hidden_states = block(hidden_states, *args, **kwargs)
                        return hidden_states
                    elif block_idx in self.uncontinuous1:
                        hidden_states += self.delta_cache.get(block_idx)
                        return hidden_states
                    
                    elif block_idx in self.end1:
                        hidden_states += self.delta_cache.get(block_idx)
                        return hidden_states
                    
                    elif block_idx in self.continuous_numbers1:
                        return hidden_states
                    else:
                        hidden_states = block(hidden_states, *args, **kwargs)
                        return hidden_states
                        # print(len(self.uncontinguous_cache))
        
            elif step_idx < self.step_start:
                hidden_states = block(hidden_states, *args, **kwargs)
                return hidden_states
        else:       #均匀step wise
            if step_idx >= self.step_start:
                if (step_idx - self.step_start) % self.step_interval == 0:
                    
                    if block_idx in self.uncontinuous: #[0, 2, [4, 7], [10,21], 23, [25, 27], 31, 33]
                        prev_hidden_states = hidden_states.clone()
                        hidden_states = block(hidden_states, *args, **kwargs)
                        delta = hidden_states - prev_hidden_states
                        self.delta_cache[block_idx] = delta
                        return hidden_states
                    elif block_idx in self.start:
                        
                        hidden_states = block(hidden_states, *args, **kwargs)
                        self.prev_hidden = hidden_states.clone()
                        return hidden_states
                    
                    elif block_idx in self.end:
                        hidden_states = block(hidden_states, *args, **kwargs)
                        cache = hidden_states - self.prev_hidden
                        self.delta_cache[block_idx] = cache
                        return hidden_states
                    else:
                        hidden_states = block(hidden_states, *args, **kwargs)
                        return hidden_states
                else:
                    if block_idx in self.missing_numbers: 
                        hidden_states = block(hidden_states, *args, **kwargs)
                        return hidden_states
                    elif block_idx in self.uncontinuous:
                        hidden_states += self.delta_cache.get(block_idx)
                        return hidden_states
                    
                    elif block_idx in self.end:
                        hidden_states += self.delta_cache.get(block_idx)
                        return hidden_states
                    
                    elif block_idx in self.continuous_numbers:
                        return hidden_states
                    else:
                        hidden_states = block(hidden_states, *args, **kwargs)
                        return hidden_states
                        # print(len(self.uncontinguous_cache))
        
                    
            elif step_idx < self.step_start: #Before step_start
                hidden_states = block(hidden_states, *args, **kwargs)
                return hidden_states
        
    def forward_single(self, block, step_idx, block_idx, hidden_states, *args, **kwargs):
        """
        Cache foreground and background blocks alternately every few steps
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
        sample_optim_flag = False
        # Before step_start: compute normally.
        if sample_optim_flag:
            if step_idx >= self.step_start:
                # if step_idx in self.step_cache_list:
                if step_idx <=31:
                    if step_idx in [6, 26, 38, 46]:
                    # if step_idx in [6, 8, 12, 18, 26, 38]:
                        # h_s = h_s + \Delta_{\text{cache}}^{(i)}
                        if block_idx in self.uncontinuous1: #[0, 2, [4, 7], [10,21], 23, [25, 27], 31, 33]
                            prev_hidden_states = hidden_states.clone()
                            hidden_states = block(hidden_states, *args, **kwargs)
                            delta = hidden_states - prev_hidden_states
                            self.delta_cache[block_idx] = delta
                            return hidden_states
                        elif block_idx in self.start1:
                            
                            hidden_states = block(hidden_states, *args, **kwargs)
                            self.prev_hidden = hidden_states.clone()
                            return hidden_states
                        
                        elif block_idx in self.end1:
                            hidden_states = block(hidden_states, *args, **kwargs)
                            cache = hidden_states - self.prev_hidden
                            self.delta_cache[block_idx] = cache
                            return hidden_states
                        else:
                            hidden_states = block(hidden_states, *args, **kwargs)
                            return hidden_states
                    else:
                        if block_idx in self.missing_numbers1: 
                            hidden_states = block(hidden_states, *args, **kwargs)
                            return hidden_states
                        elif block_idx in self.uncontinuous1:
                            hidden_states += self.delta_cache.get(block_idx)
                            return hidden_states
                        
                        elif block_idx in self.end1:
                            hidden_states += self.delta_cache.get(block_idx)
                            return hidden_states
                        
                        elif block_idx in self.continuous_numbers1:
                            return hidden_states
                        else:
                            hidden_states = block(hidden_states, *args, **kwargs)
                            return hidden_states
                            # print(len(self.uncontinguous_cache))
        
                else:
                    if step_idx in [18, 32, 42, 48]:
                        if block_idx in self.uncontinuous: #[0, 2, [4, 7], [10,21], 23, [25, 27], 31, 33]
                            prev_hidden_states = hidden_states.clone()
                            hidden_states = block(hidden_states, *args, **kwargs)
                            delta = hidden_states - prev_hidden_states
                            self.delta_cache[block_idx] = delta
                            return hidden_states
                        elif block_idx in self.start:
                            
                            hidden_states = block(hidden_states, *args, **kwargs)
                            self.prev_hidden = hidden_states.clone()
                            return hidden_states
                        
                        elif block_idx in self.end:
                            hidden_states = block(hidden_states, *args, **kwargs)
                            cache = hidden_states - self.prev_hidden
                            self.delta_cache[block_idx] = cache
                            return hidden_states
                        else:
                            hidden_states = block(hidden_states, *args, **kwargs)
                            return hidden_states
                    else:
                        if block_idx in self.missing_numbers: 
                            hidden_states = block(hidden_states, *args, **kwargs)
                            return hidden_states
                        elif block_idx in self.uncontinuous:
                            hidden_states += self.delta_cache.get(block_idx)
                            return hidden_states
                        
                        elif block_idx in self.end:
                            hidden_states += self.delta_cache.get(block_idx)
                            return hidden_states
                        
                        elif block_idx in self.continuous_numbers:
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
                    
                    if block_idx in self.uncontinuous: #[0, 2, [4, 7], [10,21], 23, [25, 27], 31, 33]
                        prev_hidden_states = hidden_states.clone()
                        hidden_states = block(hidden_states, *args, **kwargs)
                        delta = hidden_states - prev_hidden_states
                        self.delta_cache[block_idx] = delta
                        return hidden_states
                    elif block_idx in self.start:
                        
                        hidden_states = block(hidden_states, *args, **kwargs)
                        self.prev_hidden = hidden_states.clone()
                        return hidden_states
                    
                    elif block_idx in self.end:
                        hidden_states = block(hidden_states, *args, **kwargs)
                        cache = hidden_states - self.prev_hidden
                        self.delta_cache[block_idx] = cache
                        return hidden_states
                    else:
                        hidden_states = block(hidden_states, *args, **kwargs)
                        return hidden_states
                else:
                    if block_idx in self.missing_numbers: 
                        hidden_states = block(hidden_states, *args, **kwargs)
                        return hidden_states
                    elif block_idx in self.uncontinuous:
                        hidden_states += self.delta_cache.get(block_idx)
                        return hidden_states
                    
                    elif block_idx in self.end:
                        hidden_states += self.delta_cache.get(block_idx)
                        return hidden_states
                    
                    elif block_idx in self.continuous_numbers:
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
