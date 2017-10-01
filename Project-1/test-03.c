#include <stdio.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>

void process1();
void process2();
void process3();
void process4();

int main() {

	int pid1 = -10, pid2 = -10, pid3 = -10, pid4 = -10;
	int id = shmget(6969, sizeof(int), IPC_CREAT | 0666);

	if(id < 0) {
		fprintf(stderr, "Creating shared memory failed.\n");
		return 1;
	}

	pid1 = fork();

	if(pid1 > 0)
		pid2 = fork();
	
	if(pid1 > 0 && pid2 > 0)
		pid3 = fork();

	if(pid1 == 0) {
		printf("Child 1 pid1 = %d, pid2 = %d, pid3 = %d\n", pid1, pid2, pid3);
		_exit(0);
	}
	if(pid2 == 0) {
		printf("Child 2 pid1 = %d, pid2 = %d, pid3 = %d\n", pid1, pid2, pid3);
		_exit(0);
	}
	if(pid3 == 0) {
		printf("Child 3 pid1 = %d, pid2 = %d, pid3 = %d\n", pid1, pid2, pid3);
		_exit(0);
	}
	
	//if(pid1 && pid2 && pid3)
	printf("\nParent pid1 = %d, pid2 = %d, pid3 = %d\n", pid1, pid2, pid3);

	return 0;
}