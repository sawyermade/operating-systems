#include <stdio.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <sys/sem.h>

//sem shit
int semid;
struct sembuf buf[2];

//4 process function protos
void process1(int* shm);
void process2(int* shm);
void process3(int* shm);
void process4(int* shm);

int main() {

	//vars for process ids 1-4 and shared mem id and pointer
	int pid1, pid2, pid3, pid4, shmid, *shm;
	//key_t key = 6969;

	//gets shared memory, errors if unsuccessful
	shmid = shmget(IPC_PRIVATE, sizeof(int), IPC_CREAT | 0666);
	if(shmid < 0) {
		fprintf(stderr, "\nCreating shared memory failed.\n");
		return 1;
	}

	//attaches shared mem to parent process. erorrs if failed.
	shm = shmat(shmid, NULL, 0);
	if(shm == (int*)-1) {
		fprintf(stderr, "\nAssigning shared memory failed.\n");
		return 1;
	}
	
	//sem shit
	semid = semget(1234, 1, IPC_CREAT | 0666);

	//creates 4 child processes and runs the according function.
	(pid1 = fork() == 0) ? process1(shm) : NULL;
	(pid2 = fork() == 0) ? process2(shm) : NULL;
	(pid3 = fork() == 0) ? process3(shm) : NULL;
	(pid4 = fork() == 0) ? process4(shm) : NULL;

	//another way to wait for the children to finish
	// pid1 = waitpid(pid1, NULL, 0);
	// printf("Child with ID: %d has just exited.\n", pid1);

	// pid2 = waitpid(pid2, NULL, 0);
	// printf("Child with ID: %d has just exited.\n", pid2);

	// pid3 = waitpid(pid3, NULL, 0);
	// printf("Child with ID: %d has just exited.\n", pid3);

	// pid4 = waitpid(pid4, NULL, 0);
	// printf("Child with ID: %d has just exited.\n", pid4);
	
	//parent waits for children to end and prints stuff.
	pid1 = wait(&pid1);
		printf("Child with ID: %d has just exited.\n", pid1);
	pid2 = wait(&pid2);
		printf("Child with ID: %d has just exited.\n", pid2);
	pid3 = wait(&pid3);
		printf("Child with ID: %d has just exited.\n", pid3);
	pid4 = wait(&pid4);
		printf("Child with ID: %d has just exited.\n", pid4);

	//detaches and clears shared memory
	shmdt(shm);
	shmctl(shmid, IPC_RMID, NULL);

	//sem shit
	semctl(semid, 0, IPC_RMID);

	//end of program
	printf("\nEnd of Program\n");
	return 0;
}

//all 4 child processes increment the shared value a set amount of times
//then print count and detach shared memory and exit the process
void process1(int* shm) {

	int i = 0;
	while(i < 100000) {
		
		buf[0].sem_num = 0;
		buf[0].sem_op = 0;
		buf[0].sem_flg = SEM_UNDO;

		buf[1].sem_num = 0;
		buf[1].sem_op = 1;
		buf[1].sem_flg = SEM_UNDO | IPC_NOWAIT;

		semop(semid, buf, 2);

		i++;
		*shm += 1;

		buf[0].sem_op = -1;
		buf[0].sem_flg = SEM_UNDO | IPC_NOWAIT;
		semop(semid, buf, 1);
	}

	printf("From Process 1: counter = %d\n", *shm);

	shmdt(shm);
	_exit(0);
}

void process2(int* shm) {
	
	int i = 0;
	while(i < 200000) {
		
		buf[0].sem_num = 0;
		buf[0].sem_op = 0;
		buf[0].sem_flg = SEM_UNDO;

		buf[1].sem_num = 0;
		buf[1].sem_op = 1;
		buf[1].sem_flg = SEM_UNDO | IPC_NOWAIT;

		semop(semid, buf, 2);

		i++;
		*shm += 1;

		buf[0].sem_op = -1;
		buf[0].sem_flg = SEM_UNDO | IPC_NOWAIT;
		semop(semid, buf, 1);
	}

	printf("From Process 2: counter = %d\n", *shm);

	shmdt(shm);
	_exit(0);
}

void process3(int* shm) {

	int i = 0;
	while(i < 300000) {
		
		buf[0].sem_num = 0;
		buf[0].sem_op = 0;
		buf[0].sem_flg = SEM_UNDO;

		buf[1].sem_num = 0;
		buf[1].sem_op = 1;
		buf[1].sem_flg = SEM_UNDO | IPC_NOWAIT;

		semop(semid, buf, 2);

		i++;
		*shm += 1;

		buf[0].sem_op = -1;
		buf[0].sem_flg = SEM_UNDO | IPC_NOWAIT;
		semop(semid, buf, 1);
	}

	printf("From Process 3: counter = %d\n", *shm);

	shmdt(shm);
	_exit(0);
}

void process4(int* shm) {

	int i = 0;
	while(i < 500000) {
		
		buf[0].sem_num = 0;
		buf[0].sem_op = 0;
		buf[0].sem_flg = SEM_UNDO;

		buf[1].sem_num = 0;
		buf[1].sem_op = 1;
		buf[1].sem_flg = SEM_UNDO | IPC_NOWAIT;

		semop(semid, buf, 2);

		i++;
		*shm += 1;

		buf[0].sem_op = -1;
		buf[0].sem_flg = SEM_UNDO | IPC_NOWAIT;
		semop(semid, buf, 1);
	}
	
	printf("From Process 4: counter = %d\n", *shm);

	shmdt(shm);
	_exit(0);
}