#include <stdio.h>

int main() {
	
	int x = 10, y = 10, count = 1;
	while(count < 3) {

		if(x != 0) {
			x = fork();
			y += 10;
		}
		else {
			x = fork();
			y += 50;
		}
		printf("y = %d\n", y);
		++count;
	}
}

//(888) 637-2176