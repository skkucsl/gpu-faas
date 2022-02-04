#include <stdio.h>

struct test {
	int arr[100000];
};

int main(){
	//struct test t[100];
	int i,j;
	int a[1000000];
	for(i=0;i<1000000;i++) a[i]=i;
	printf("int array done\n");
	/*
	for(i=0;i<100;i++)
		for(j=0;j<100000;j++)
			t[i].arr[j]=j;
			*/
	for(;;);
	return 0;
}
