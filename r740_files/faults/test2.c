#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/types.h>
#include <linux/kernel.h>
#include <sys/syscall.h>
#include <sys/mman.h>

int main() {
	int x;
	long tmp;
	int *arr;
	char *file = NULL;
	int fd = open("hello.txt", O_RDONLY);

	//file = (char *)mmap(0, 1024*4*1000*10, PROT_WRITE | PROT_READ, MAP_PRIVATE, fd, 0);
	//syscall(548, 1);
	
	arr = (int *)mmap(0, sizeof(int)*1024*1000*100, PROT_WRITE | PROT_READ, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);
	/* Anon read fault */
/*
	for (int i=0; i<1000*100; i++)
		printf("%d ", arr[i*1024]);

	printf("\n");
*/
	/* Anon write fault if no read, cow fault if yes read */
/*
	for (int i=0; i<1000*100; i++)
		arr[i*1024] = i+1;
*/	
	file = (char *)mmap(0, 1024*1000*100, PROT_WRITE | PROT_READ, MAP_PRIVATE, fd, 0);
	/* File read fault */
	/* File write fault if no write */

	for (int i=0; i<10000; i++)
                printf("%c ", file[i*4096]);
	printf("\n");


	for (int i=0; i<10000; i++)
		file[i*4096] = 'x';

	munmap(file, 1024*100*1000);
	munmap(arr, sizeof(int)*100*1000*1024);
	close(fd);

//	usleep(1000*1000);

//	syscall(551);
//	syscall(549, 2);

//	printf("Hello?\n");
	
	return 0;
}
