//#define _REENTRANT
#include <pthread.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>
#include <stdlib.h>

//constants/globals
#define BUFFSIZE 15

//pthread parameter struct
typedef struct ptparam {
	sem_t mutex, full, empty;
	char buff[BUFFSIZE];
	FILE *fp;
} ptparam;

//functions
void *producer(void *buff);
void *consumer(void *buff);

int main() {

	//file open stuff
	FILE *fp = fopen("mytest.dat", "r");
	if(!fp) {
		printf("\nCould not open file mytest.dat\n");
		return -1;
	}

	//shared memory parameter struct for threads
	ptparam *buff;

	//creates shared mem
	int shmid = shmget(IPC_PRIVATE, sizeof(ptparam), IPC_CREAT | 0666);
	if(shmid < 0) {
		printf("\nCreating shared memory failed.\n");
		return -1;
	}

	//attaches shared mem to parent process. erorrs if failed.
	buff = shmat(shmid, NULL, 0);
	if(buff == (ptparam*)-1) {
		printf("\nAssigning shared memory failed.\n");
		return -1;
	}

	//sets file pointer into shared mem
	buff->fp = fp;

	//semaphore init
	sem_init(&buff->mutex, 0, 1);
	sem_init(&buff->full, 0, 0);
	sem_init(&buff->empty, 0, BUFFSIZE);

	//thread vars
	pthread_t pro, con;
	pthread_attr_t attr;

	//setup threads
	fflush(stdout);
	pthread_attr_init(&attr);
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);

	//creates threads and waits for them to finish
	pthread_create(&pro, &attr, producer, buff);
	pthread_create(&con, &attr, consumer, buff);
	pthread_join(pro, NULL);
	pthread_join(con, NULL);

	//destroy semaphores
	sem_destroy(&buff->mutex);
	sem_destroy(&buff->full);
	sem_destroy(&buff->empty);

	//detaches and clears shared memory
	shmdt(buff);
	if(shmctl(shmid, IPC_RMID, NULL) < 0)
		printf("\nRemoving shared memory ID %d failed.\n", shmid);

	//closes file and ends
	fclose(fp);
	printf("\n");
	return 0;
}

//producer function
void *producer(void *buff) {

	int i = 0, stop = 0;
	char c;
	while(i++ < 150 && (c = getc(((ptparam*)buff)->fp)) != EOF) {

		sem_wait(&((ptparam*)buff)->empty);
		sem_wait(&((ptparam*)buff)->mutex);

		((ptparam*)buff)->buff[i%BUFFSIZE] = c;

		sem_post(&((ptparam*)buff)->mutex);
		sem_post(&((ptparam*)buff)->full);
	}

	sem_wait(&((ptparam*)buff)->empty);
	sem_wait(&((ptparam*)buff)->mutex);

	((ptparam*)buff)->buff[i%BUFFSIZE] = '*';

	sem_post(&((ptparam*)buff)->mutex);
	sem_post(&((ptparam*)buff)->full);
}

//consumer function
void *consumer(void *buff) {

	int i = 0, stop = 0;
	char c = '0';
	while(i++ < 150 && c != '*') {

		sem_wait(&((ptparam*)buff)->full);
		sem_wait(&((ptparam*)buff)->mutex);

		c = ((ptparam*)buff)->buff[i%BUFFSIZE];
		if(c != '*')
			printf("%c", c);

		sem_post(&((ptparam*)buff)->mutex);
		sem_post(&((ptparam*)buff)->empty);
	}
}