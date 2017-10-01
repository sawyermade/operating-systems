#include <sys/types.h>
#include <stdio.h>
#include <unistd.h>

int main() {

	//pid_t pid;
	int pid;

	//creates child process
	pid = fork();

	if(pid < 0) {

		fprintf(stderr, "Fork Failed\n");
		return 1;
	}

	else if(pid == 0) {
		execlp("/bin/ls", "ls", "-l", NULL);
	}

	else {
		wait(NULL);
		printf("\nChild Process: %d Complete\n", pid);
	}

	return 0;
}