import pycuda.driver as cuda
import pycuda.autoinit

def get_gpu_info(device_id=0):
    device = cuda.Device(device_id)
    gpu_info = {
        "name": device.name(),
        "compute_capability": device.compute_capability(),
        "total_memory": device.total_memory() / (1024 * 1024),
        "clock_rate": device.get_attribute(cuda.device_attribute.CLOCK_RATE) / 1000,
        "memory_clock_rate": device.get_attribute(cuda.device_attribute.MEMORY_CLOCK_RATE) / 1000,
        "num_multiprocessors": device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT),
        "max_threads_per_block": device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK),
        "max_threads_per_multiprocessor": device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR),
        "warp_size": device.get_attribute(cuda.device_attribute.WARP_SIZE),
        "l2_cache_size": device.get_attribute(cuda.device_attribute.L2_CACHE_SIZE),
        "max_shared_memory_per_multiprocessor": device.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR),
        "global_memory_bus_width": device.get_attribute(cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH),
    }
    
    # Write to CSV
    # filename = f"{gpu_info['name']}_gpu_info.csv"
    # with open(filename, 'w') as f:
    #     writer = csv.writer(f)
    #     for key, value in gpu_info.items():
    #         writer.writerow([key, value])
    return gpu_info
        
if __name__ == "__main__":
    get_gpu_info()
