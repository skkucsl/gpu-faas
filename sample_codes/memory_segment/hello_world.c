#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(){
	FILE *fp;
	char buf[1024], sbuf[102400];;//, ebuf[102400];
	
	
	fp = popen("/usr/bin/pmap -x `cut -d ' ' -f 4 /proc/self/stat`", "r");
	memset(sbuf, 0, 102400);
	while(fgets(buf, sizeof(buf), fp) != NULL){
		printf("%s", buf);
		strcat(sbuf, buf);
	}
	printf("{\'pmap_result\': \'%s\'}", sbuf);
	return 0;
}
