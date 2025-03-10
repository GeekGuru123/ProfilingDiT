class DitCache:
    def __init__(self, step_start, step_interval, block_start, num_blocks, block_start_double=None,
                 num_blocks_double=None,  **kwargs):
        self.step_start = step_start
        self.step_interval = step_interval
        self.block_start = block_start
        self.num_blocks = num_blocks
        self.block_end = block_start + num_blocks - 1
        # double blocks
        self.block_start_double = block_start_double
        self.num_blocks_double = num_blocks_double
        self.block_end_double = block_start_double + num_blocks_double - 1 if block_start_double is not None else None
        self.cache = 0
        self.time_cache = {}
    
    def forward_single(self, block, time_step, block_idx, hidden_states, *args, **kwargs
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
        if not self.use_cache(time_step, block_idx):
            old_hidden_states = hidden_states
            hidden_states = block(hidden_states, *args, **kwargs)
            self.update_cache(hidden_states, old_hidden_states, time_step, block_idx)
        else:
            hidden_states += self.get_cache(time_step, block_idx)
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