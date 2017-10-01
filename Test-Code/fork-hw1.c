#include <stdio.h>

int main() {
	
	int m = 10, n = 5, count = 1, mult = 1;

	while(count < 3) {

		if(m != 0) {
			m = fork();
			n += 25;
		}

		else {
			m = fork();
			n += 20;
			mult *= n;
		}

		printf("n = %d	mult = %d\n", n, mult);
		++count;
	}
}