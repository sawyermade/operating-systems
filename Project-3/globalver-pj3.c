#define _REENTRANT
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
#define SIZE 15
sem_t mutex, full, empty;
char *buff;
FILE *fp;

//functions
void *producer(void *parm);
void *consumer(void *parm);

int main() {

	//file open stuff
	fp = fopen("mytest.dat", "r");

	//creates shared mem
	int shmid = shmget(IPC_PRIVATE, sizeof(char)*SIZE, IPC_CREAT | 0666);

	//attaches shared mem to parent process. erorrs if failed.
	buff = shmat(shmid, NULL, 0);

	//semaphore init
	sem_init(&mutex, 0, 1);
	sem_init(&full, 0, 0);
	sem_init(&empty, 0, SIZE);

	//thread vars
	pthread_t pro, con;
	pthread_attr_t attr;

	//setup threads
	fflush(stdout);
	pthread_attr_init(&attr);
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);

	//creates threads and waits for them to finish
	pthread_create(&pro, &attr, producer, buff);
	sleep(1);
	pthread_create(&con, &attr, consumer, buff);
	pthread_join(pro, NULL);
	pthread_join(con, NULL);

	//destroy semaphores
	sem_destroy(&mutex);
	sem_destroy(&full);
	sem_destroy(&empty);

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
void *producer(void *parm) {

	int i = 0;
	char c;
	while(i++ < 150 && (c = getc(fp)) != EOF) {

		sem_wait(&empty);
		sem_wait(&mutex);

		buff[i%SIZE] = c;

		sem_post(&mutex);
		sem_post(&full);
	}

	sem_wait(&empty);
	sem_wait(&mutex);

	buff[i%SIZE] = '*';

	sem_post(&mutex);
	sem_post(&full);
}

//consumer function
void *consumer(void *parm) {

	int i = 0;
	char c = '0';
	while(i++ < 150 && c != '*') {

		sem_wait(&full);
		sem_wait(&mutex);

		c = buff[i%SIZE];
		if(c != '*')
			printf("%c", c);
		
		sem_post(&mutex);
		sem_post(&empty);
	}
}