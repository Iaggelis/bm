#include "./bm.h"

#include <fftw3.h>


Err bm_fftw_alloc_real(Bm *bm);
Err bm_fftw_alloc_complex(Bm *bm);
Err bm_fftw_plan_dft_r2c_1d(Bm *bm);
Err bm_fftw_plan_dft_c2r_1d(Bm *bm);

Err bm_fftw_alloc_real(Bm *bm)
{
    uint64_t n = bm->stack[bm->stack_size - 1].as_u64;
    double *input_buffer  = fftw_alloc_real(n);

    printf("[DBG] creating array of doubles:\n");
    for(size_t i = 0; i < n; ++i) {
        input_buffer[i] = bm->stack[bm->stack_size - 1  - n + i].as_f64;
        printf("[DBG] %f\n", input_buffer[i]);
    }

    bm->stack[bm->stack_size++].as_ptr = input_buffer;

    return ERR_OK;
}

Err bm_fftw_alloc_complex(Bm *bm)
{
    uint64_t n = 2 * bm->stack[bm->stack_size - 1].as_u64;
    fftw_complex *input_buffer  = fftw_alloc_complex(n);

    for(size_t i = 0; i < n; i+=2) {
        input_buffer[i][0] = bm->stack[bm->stack_size - 1  - n + i].as_f64;
        input_buffer[i][1] = bm->stack[bm->stack_size - 1  - n + i + 1].as_f64;
        printf("[DBG] %f, %f\n", input_buffer[i][0], input_buffer[i][1]);
    }

    bm->stack[bm->stack_size - 1].as_u64 = (uint64_t) input_buffer;

    return ERR_OK;
}

Err bm_fftw_plan_dft_r2c_1d(Bm *bm)
{

    if (bm->stack_size < 3) {
        return ERR_STACK_UNDERFLOW;
    }

    double *x = (double*) bm->stack[bm->stack_size - 3].as_u64;
    int n = (int) bm->stack[bm->stack_size - 2].as_i64;
    uint32_t mode = (uint32_t) bm->stack[bm->stack_size - 1].as_u64;

    size_t fft_out_size = (size_t) (n/2 + 1);
    fftw_complex *output_buffer = fftw_alloc_complex(fft_out_size);

    printf("[DBG] running fft\n");
    fftw_plan p = fftw_plan_dft_r2c_1d(n, x, output_buffer, mode);

    fftw_execute(p);

    printf("[DBG] output complex array\n");
    for(size_t i = 0; i < fft_out_size; ++i) {
        printf("[DBG] %f, %fj\n", output_buffer[i][0], output_buffer[i][1]);
    }
    fftw_destroy_plan(p);

    bm->stack[bm->stack_size - 1].as_u64 = (uint64_t) output_buffer;

    return ERR_OK;
}

Err bm_fftw_plan_dft_c2r_1d(Bm *bm)
{

    if (bm->stack_size < 3) {
        return ERR_STACK_UNDERFLOW;
    }

    fftw_complex *x = (fftw_complex*) bm->stack[bm->stack_size - 3].as_ptr;
    int n = (int) bm->stack[bm->stack_size - 2].as_i64;
    uint32_t mode = (uint32_t) bm->stack[bm->stack_size - 1].as_u64;

    size_t fft_out_size = (size_t) (2 * (n - 1));
    double *output_buffer = fftw_alloc_real(fft_out_size);

    printf("[DBG] running fft\n");
    fftw_plan p = fftw_plan_dft_c2r_1d(n, x, output_buffer, mode);

    fftw_execute(p);

    printf("[DBG] But to reality\n");
    for(size_t i = 0; i < fft_out_size; ++i) {
        // FFTW scales the output so we need to divide by the length
        output_buffer[i] /= (double) n;
        printf("[DBG] %f\n", output_buffer[i]);
    }
    fftw_destroy_plan(p);

    bm->stack[bm->stack_size - 1].as_u64 = (uint64_t) output_buffer;

    return ERR_OK;
}
