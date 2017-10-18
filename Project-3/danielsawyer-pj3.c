#define _REENTRANT
#include <pthread.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <semaphore.h>

#define BUFFSIZE 15

//pthread parameter struct
typedef struct ptparam {
	sem_t mutex, full, empty;
	char *shmbuf;
	int shmid;
} ptparam;

int main() {

	//
	FILE *fp;
	fp = fopen("mytest.dat", "r");
	if(!fp) {
		printf("\nCould not open file mytest.dat\n");
		return -1;
	}

	//shared memory parameter struct
	ptparam buff;

	//
	buff.shmid = shmget(IPC_PRIVATE, sizeof(char)*BUFFSIZE, IPC_CREAT | 0666);
	if(buff.shmid < 0) {
		printf("\nCreating shared memory failed.\n");
		return -1;
	}

	//attaches shared mem to parent process. erorrs if failed.
	buff.shmbuf = shmat(buff.shmid, NULL, 0);
	if(buff.shmbuf == (char*)-1) {
		printf("\nAssigning shared memory failed.\n");
		return -1;
	}

	//semaphore init
	sem_init(&buff.mutex, 0, 1);
	sem_init(&buff.full, 0, 0);
	sem_init(&buff.empty, 0, BUFFSIZE);

	pthread_t tid1, tid2;
	pthread_attr_t attr;

	fflush(stdout);

	pthread_attr_init(&attr);
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);



	//detaches and clears shared memory
	shmdt(buff.shmbuf);
	if(shmctl(buff.shmid, IPC_RMID, NULL) < 0) {
		printf("\nRemoving shared memory ID %d failed.\n", buff.shmid);
		//exit(EXIT_FAILURE);
	}

	//destroy semaphores
	sem_destroy(&buff.mutex);
	sem_destroy(&buff.full);
	sem_destroy(&buff.empty);

	return 0;
}