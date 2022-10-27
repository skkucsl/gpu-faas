#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <linux/kernel.h>
#include <sys/types.h>
#include <sys/syscall.h>

int main() {
	long tmp;
	//int *arr;
	pid_t p;
	char *binpath = "/home/user01/r740_files/kernel_test/faults/b.out";
	char *args[] = {binpath, "b.out", 0}; 

	tmp = syscall(552);
	tmp = syscall(550);
	
	if (fork() > 0) {
		printf("I'm parent(%d), wake up after 1 seconds\n", getpid());
		usleep(1000*1000);
	}
	else {
		execv(binpath, args);
	}
/*
	arr = (int*)malloc(sizeof(int)*1024*1024);
	for(int i=0; i<1024*1024; i++)
		arr[i] = i;
	free(arr);
*/	
	return 0;
}
