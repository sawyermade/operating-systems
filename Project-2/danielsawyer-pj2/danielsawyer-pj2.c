#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <sys/sem.h>

//semaphore union typedef
typedef union{
	int val;
	struct semid_ds *buf;
	ushort *array;
} semunion;

//global vars
int semid;
static struct sembuf OP = {0,-1,0}; //waits and locks values
static struct sembuf OV = {0,1,0}; //unlocks values
struct sembuf *P = &OP, *V = &OV; //ptrs for above 2
#define SEMKEY (key_t)400L //key for semget()
#define NSEMS 1 //number of semaphores

//functions
void process(int* shm, int pnum, int increment); //process func
int POP(){int status; status = semop(semid, P, 1); return status;} //waits and locks func
int VOP(){int status; status = semop(semid, V, 1); return status;} //unlocks func


int main() {

	//vars for process ids 1-4 and shared mem id and pointer
	int pid1, pid2, pid3, pid4, shmid, *shm;

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
	
	//gets semaphore and sets val to 1
	semid = semget(SEMKEY, NSEMS, IPC_CREAT | 0666);
	if(semid == -1) {
		fprintf(stderr, "\nAssigning semaphore failed.\n");
		return 1;
	}
	semunion arg;
	arg.val = 1;
	semctl(semid, 0, SETVAL, arg);

	//creates 4 child processes and runs the func with proper values.
	(pid1 = fork() == 0) ? process(shm, 1, 100000) : NULL;
	(pid2 = fork() == 0) ? process(shm, 2, 200000) : NULL;
	(pid3 = fork() == 0) ? process(shm, 3, 300000) : NULL;
	(pid4 = fork() == 0) ? process(shm, 4, 500000) : NULL;
	
	//parent waits for children to end and prints stuff.
	printf("Child with ID: %d has just exited.\n", wait(&pid1));
	printf("Child with ID: %d has just exited.\n", wait(&pid2));
	printf("Child with ID: %d has just exited.\n", wait(&pid3));
	printf("Child with ID: %d has just exited.\n", wait(&pid4));

	//detaches and clears shared memory
	shmdt(shm);
	shmctl(shmid, IPC_RMID, NULL);

	//removes semaphore
	semctl(semid, 0, IPC_RMID);

	//end of program
	printf("\nEnd of Program\n");
	return 0;
}

//runs child processes with protected shm
void process(int* shm, int pnum, int increment) {

	int i = 0;
	while(i++ < increment) {
		POP();
		*shm += 1;
		VOP();
	}

	printf("From Process %d: counter = %d\n", pnum, *shm);
	shmdt(shm);
	exit(0);
}
