#include <iostream>
#include <CL/sycl.hpp>
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string>
#include "filter.h"
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif
using namespace std;
using namespace sycl;


#include <vector>
static auto exception_handler = [](sycl::exception_list e_list) {
	for (std::exception_ptr const& e : e_list) {
		try {
			std::rethrow_exception(e);
		}
		catch (std::exception const& e) {
#if _DEBUG
			std::cout << "Failure" << std::endl;
#endif
			std::terminate();
		}
	}
};

// the following three has to be decided by us regarding input size









// decide on how many zeroes to append
int Initialise(int BlockSize) {

	// append M-1 zeroes to X(n) sections
	// M + M -1 = filter_length*2 -1, therefore 2x21-1 = 41
	/*******************************************Step I*************************************************/
	// filter size is size H(n)
	// determine M which is the filter size
	// then N = M+L-1, so L = N-M+13
	int L = BlockSize - filter_length + 1;

	/*******************************************Step II*************************************************/
	// append L-1 zeroes to h(n)
	int newSize = L - 1;
	return newSize;
}


// while in the FPGA node...
// perform convolution

int conv(queue &q, int a, int num_batches,int nf, int ng, int n, double *Xn, double *output,double *Hn, double *chunk,
	double *output_head, double *output_tail, double *output_overlap, double *previous, double *out) {
	// end result length
	
	range<1> num_items{ 81 };
	range<1> num_iters{ 41 };
	auto e = q.parallel_for(num_items, [=](auto i) {
		int const jmn = (i >= ng - 1) ? i - (ng - 1) : 0;
		int const jmx = (i < nf - 1) ? i : nf - 1;
		for (auto j(jmn); j <= jmx; ++j) {
			out[i] = (chunk[j] * Hn[i - j]);
		}
		});
	e.wait();
	

	// return the appropriate vector according to current iteration
	if (a == 0) {
		// first chunk, simply send back the first 20 elements (Yn_head)
		auto k = q.parallel_for(num_iters, [=](auto b) {
			if (b < 20) {
				output_head[b] = out[b];
			}
			// afterwards, replace prev with current
			previous[b] = out[b];
			});
		k.wait();
		
		/************************ send back out_head ******************************/
		return 1;
	}
	else if ((a != 0) && (a != num_batches)) {
		// from second chunk onward we only send overlapped sections
		auto r = q.parallel_for(num_iters, [=](auto b) {
			if (b == 0) {
				output_overlap[0] = previous[20];

				// cout << "index: " << b - 20 << "\n";
				// cout << "first one:" << Yn_FPGA_prev[b] << "\n";
			}
			else {
				output_overlap[b] = previous[b+20] + out[b-1];
			}
			});
		r.wait();
		
			// cout << "overlap: " << Yn_overlap[b-20] << "\n";
		
		for (int c = 0; c < 41; c++) {
			previous[c] = out[c];

			// cout << "      current: " << Yn_FPGA_current[c] << "\n";
		}
		/************************ send back out_overlap ******************************/
		return 2;
	}
	else {
		// last chunk, we send both the overlap and the tail
		
		auto u = q.parallel_for(num_iters, [=](auto b) {
			if (b == 0) {
				output_overlap[0] = previous[20];
			}
			else {
				output_overlap[b] = previous[20] + out[b-1];
			}
			// non-overlapping end section
			output_tail[b] = out[b+20];
			});
		u.wait();
		/****************** send back out_overlap and out_tail ************************/
		return 3;
	}
	// return out;
}


// main function here
int main()
{
#if FPGA_EMULATOR
	// DPC++ extension: FPGA emulator selector on systems without FPGA card.
	ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
	// DPC++ extension: FPGA selector on systems with FPGA card.
	ext::intel::fpga_selector d_selector;
#else
	// The default device selector will select the most performant device.
	default_selector d_selector;
#endif
	size_t input_size = 210;
	size_t block_size = 41;
	size_t filter_length = 21;
	size_t na = 81;
	int num_batches = 10;
	int append = 20;
	try {
		queue q(d_selector, exception_handler);
		double* Xn = malloc_shared<double>(input_size, q);
		
		double* Hn = malloc_shared<double>(block_size, q);
		double* chunk = malloc_shared<double>(block_size, q);
		double* output = malloc_shared<double>(input_size, q);
		double* out = malloc_shared<double>(input_size, q);
		double* previous = malloc_shared<double>(block_size, q);
		double* output_head = malloc_shared<double>(filter_length, q);
		double* output_overlap = malloc_shared<double>(filter_length, q);
		double* output_tail = malloc_shared<double>(filter_length, q);


		for (int i = 0; i < 210; i++) {
			Xn[i] += 5 * sin(2 * M_PI * 0.4 * i); // 0.4hz
			Xn[i] += 5 * sin(2 * M_PI * 70 * i); // 70hz
			Xn[i] += 10 * sin(2 * M_PI * 80 * i); // 80hz
		}

		for (int i = 0; i < block_size; i++) {
			if (i < filter_length) {
				Hn[i] = filter[i];
			}
			else {

			}

		}

		/*******************************************Step IV*************************************************/
		// compute the convolutions, chunk by chunk
		// first, form the matrix for one section of Xn
		int curr = 0;
		int offset = 0;
		int index = 0;

		for (int a = 0; a < num_batches; a++) {
			// 10 chunks in total
			// prepare a new chunk
			for (int x = 0; x < block_size; x++) {
				if (x < filter_length) {
					chunk[x] = Xn[a * filter_length + x];
				}
				else {
					chunk[x] = 0.0;
				}
			}
			int const nf = 41; // size of the input
			int const ng = 21; // size of filter
			int const n = 61;
			// invoke FPGA
			auto result = conv(q,a, num_batches,nf,ng,n,Xn,output,Hn,chunk,output_head,output_tail,output_overlap,previous,out);

			/********************************************Step V************************************************/
			// now we can process the output
			// simply concatonate the recieved output vectors together
			if (result == 1) {
				for (int i = 0; i < 20; i++) {
					output[index] = output_head[i];
					index++;
				}
			}
			else if (result == 2) {
				for (int i = 0; i < 21; i++) {
					output[index] = output_overlap[i];
					index++;
				}
			}
			else {
				for (int i = 0; i < 21; i++) {
					output[index] = output_overlap[i];
					index++;
				}
				for (int i = 0; i < 21; i++) {
					output[index] = output_tail[i];
					index++;
				}
			}
		}

		// print output for debugging
		for (int i = 0; i < sizeof(output); i++) {
			cout << output[i] << "\n";
		}
	}
	catch (std::exception const& e) {
		std::cout << "An exception is caught for FIR.\n";
		std::terminate();
	}
	return 0;
}