#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int main(int argc, char* argv[])
{
	int fd;
	char buf[1024];
	struct timeval tv;
	if (argc > 9){
		gettimeofday(&tv,NULL);
		sprintf(buf, "%s %s %s %s %s %s %s %s %s %s %d %ld\n", argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argc, tv.tv_sec*1000 + tv.tv_usec/1000);
		fd = open("/home/user01/microbench/binary_hijack/runc.txt",O_RDWR| O_CREAT,0777);
		lseek(fd,0,SEEK_END);
		write(fd,buf,strlen(buf));
		close(fd);
	}	
	execv("/usr/bin/runc.bak",argv);
} 
