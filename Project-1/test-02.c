#include <stdio.h>
int main() {
	
	int p1=1, p2=2, p3=3;
	p1 = fork();
	
	if(p2>0) p2 = fork();
	if(p1>0) p3 =fork();
	if(p1==0) printf("type 1\n");
	if(p3!=0) printf("type 2\n");
	if(p2!=0) printf("type 3\n");
	if((p1>0)||(p2>0) || (p3>0)) printf("type 4\n");
	if((p2==0) && (p3==0)) printf("type 5\n");

	return 0;
}