template <typename value_idx, typename value_t>	
struct KeyValue {	
 value_idx key;	
 value_t value;	
};	
	
const int kEmpty = 0xffffffff;	
	
// 32 bit Murmur3 hash	
	
inline __device__ uint32_t hash(uint32_t k, uint32_t capacity) {	
 k ^= k >> 16;	
 k *= 0x85ebca6b;	
 k ^= k >> 13;	
 k *= 0xc2b2ae35;	
 k ^= k >> 16;	
 return k & (capacity - 1);	
}	
	
template <typename value_idx, typename value_t>	
__device__ void gpu_hashtable_insert(KeyValue<value_idx, value_t> *hashtable,	
                                    value_idx key, value_t value,	
                                    uint32_t capacity) {	
 uint32_t slot = hash((uint32_t)key, capacity);	
	
 int n_iter = 0;	
 while (true) {	
   value_idx prev = atomicCAS(&hashtable[slot].key, kEmpty, key);	
	
   if (prev == kEmpty || prev == key) {	
     hashtable[slot].value = value;	
     printf("key added on iter: %d\n", n_iter);
     break;	
   }	
	
   slot = (slot + 1) & (capacity - 1);	
	
   // @TODO: Perform a max-reduction on this value	
   ++n_iter;	
 }	
}	
	
template <typename value_idx, typename value_t>	
__device__ value_t gpu_hashtable_lookup(KeyValue<value_idx, value_t> *hashtable,	
                                       value_idx key, uint32_t capacity) {	
 uint32_t slot = hash((uint32_t)key, capacity);	
	
 int n_iter = 0;	
	
 // @TODO: Use data to set this threshold.	
 while (n_iter < 3) {	
   if (hashtable[slot].key == key) {	
     return hashtable[slot].value;	
   }	
   if (hashtable[slot].key == kEmpty) {	
     return 0.0;	
   }	
	
   if(n_iter > 1000)	
     printf("Hash table lookup failed: bid=%d, tid=%d, slot=%d, key=%d, n_it=%d\n",	
            blockIdx.x, threadIdx.x, slot, key, n_iter);	
	
   slot = (slot + 1) & (capacity - 1);	
   ++n_iter;	
 }	
	
 return 0.0;	
}