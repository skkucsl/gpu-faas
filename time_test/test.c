#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

int main(){
		time_t start, end;
		int fd;

		start = clock();
		fd = open("./test.txt", O_RDWR | O_CREAT, 0777);
		end = clock();
		printf("open time %lf\n", (double)(end-start)/CLOCKS_PER_SEC);

		start = clock();
		write(fd, "Hi Hello World!\n", 17);
		end = clock();
		printf("write time %lf\n", (double)(end-start)/CLOCKS_PER_SEC);


		close(fd);

}
