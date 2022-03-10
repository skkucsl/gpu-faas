#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <dlfcn.h>

int main()
{
	int *temp = 0;
	clock_t start, end;

	void *handle;
	int (*test)(int);
	char *error;

	start = clock();
	cudaSetDevice(0);
	cudaMalloc((void**)&temp, sizeof(int));
	end = clock();

	printf("Wait 10 seconds\n");
	usleep(10000000);

	handle = dlopen("./libtest.so", RTLD_LAZY);
	if (!handle) {
		fprintf(stderr, "%s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	dlerror();

	*(void **)(&test) = dlsym(handle, "_Z9wait_timei");

	handle = dlopen("./libtest.so", RTLD_LAZY);
	if (!handle) {
		fprintf(stderr, "%s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	printf("%d seconds waits, main time = %lf\n", (*test)(5), (float)(end-start)/CLOCKS_PER_SEC);
	printf("Again %d seconds waits\n", (*test)(5));
	dlclose(handle);

	return 0;
}

