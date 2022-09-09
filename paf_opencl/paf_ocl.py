import numpy as np
import pyopencl as cl
import pyopencl
import time
import pandas as pd
from tqdm import tqdm
import pyopencl.cltypes as cltypes
import copy


def action(N_, iter=-1):
    prepare_data_start_time = time.time()
    prepare_data_start_time1 = time.process_time()
    N = N_
    DIM = int(N + 10)
    c = np.random.randint(low=0, high=20, size=(DIM, DIM)).astype(cltypes.char)
    F = np.random.randint(low=0, high=20, size=(DIM, DIM)).astype(cltypes.char)
    c = c.ravel()
    F = F.ravel()
    # prepare memory for final answer from OpenCL
    final = copy.deepcopy(c)

    prepare_data_stop_time = time.time()
    prepare_data_stop_time1 = time.process_time()

    # opencl
    #print('load program from cl source file')
    f = open('adjust_score.cl', 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()

    #print('create context')
    #ctx = cl.create_some_context()
    create_ctx_start_time = time.time()
    create_ctx_start_time1 = time.process_time()

    platform = pyopencl.get_platforms()[0]
    devices = platform.get_devices()
    # print(devices)
    ctx = pyopencl.Context(devices)

    create_ctx_stop_time = time.time()
    create_ctx_stop_time1 = time.process_time()

    #print('create command queue')
    ctx_queue_start_time = time.time()
    ctx_queue_start_time1 = time.process_time()

    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    ctx_queue_stop_time = time.time()
    ctx_queue_stop_time1 = time.process_time()

    #print('compile kernel code')
    compile_kernel_start_time = time.time()
    compile_kernel_start_time1 = time.process_time()
    prg = cl.Program(ctx, kernels).build()
    compile_kernel_stop_time = time.time()
    compile_kernel_stop_time1 = time.process_time()

    #print('prepare device memory for input / output')
    upload_data_start_time = time.time()
    upload_data_start_time1 = time.process_time()

    #dev_c_matrix = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=c)
    #dev_F_matrix = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=F)

    # https://documen.tician.de/pyopencl/runtime_const.html#pyopencl.mem_flags
    dev_c_matrix = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=c)
    dev_F_matrix = cl.Buffer(ctx, cl.mem_flags.READ_ONLY |
                             cl.mem_flags.COPY_HOST_PTR, hostbuf=F)
    #dev_final_matrix = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=final)

    upload_data_stop_time = time.time()
    upload_data_stop_time1 = time.process_time()

    # Obliczenia
    opencl_start = time.time()
    opencl_start1 = time.process_time()
    GRID_SIZE = 255
    evt = prg.adjust_score(queue, (255, 255), (1, 1),
                           dev_c_matrix,
                           dev_F_matrix,
                           np.int32(N),
                           )
    evt.wait()

    opencl_stop = time.time()
    opencl_stop1 = time.process_time()

    download_time_start = time.time()
    download_time_start1 = time.process_time()

    cl.enqueue_copy(queue, final, dev_c_matrix).wait()

    download_time_stop = time.time()
    download_time_stop1 = time.process_time()

    #final = final.reshape((DIM,DIM))

    d = {
        'Device name': devices[0],
        'Dimentions': N,
        'prepare data t': prepare_data_stop_time - prepare_data_start_time,
        'create ctx t': create_ctx_stop_time - create_ctx_start_time,
        'create command queue t': ctx_queue_stop_time - ctx_queue_start_time,
        'first data upload t': upload_data_stop_time - upload_data_start_time,
        'kernel compile t': compile_kernel_stop_time - compile_kernel_start_time,
        'download data time t': download_time_stop - download_time_start,
        'opencl execution time t': opencl_stop - opencl_start,

        'prepare data p': prepare_data_stop_time1 - prepare_data_start_time1,
        'create ctx p': create_ctx_stop_time1 - create_ctx_start_time1,
        'create command queue p': ctx_queue_stop_time1 - ctx_queue_start_time1,
        'first data upload p': upload_data_stop_time1 - upload_data_start_time1,
        'kernel compile p': compile_kernel_stop_time1 - compile_kernel_start_time1,
        'download data time p': download_time_stop1 - download_time_start1,
        'opencl execution time p': opencl_stop1 - opencl_start1,
    }

    #df = pd.DataFrame([d])
    # print(pd.Series(d).to_string())

    df = pd.DataFrame([d])

    def f(x):
        res = x
        try:
            res = np.round_(x, decimals=5)
        except:
            pass
        return res
    df = df.applymap(f)
    df.to_csv('czasy/ocl.csv', mode='a', index=False, header=False)

    #print(df[['Dimentions','opencl execution time t','opencl execution time p'  ]].to_string())
    # return df


if __name__ == '__main__':
    TYPE = 0

    if TYPE == 0:
        action(1000, iter=0)

    if TYPE == 1:
        iters = 20
        Ns = list(np.arange(200, 2301, 100))  # [50,100,200,500,1000, 2000]
        for i in range(iters):
            for n in Ns:
                action(n)