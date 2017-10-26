#include <stdio.h>
#include <semaphore.h>
#include <pthread.h>
#include <sys/ipc.h>
#include <sys/shm.h>
// #include <sys/wait.h>

//constants
#define BUFFSIZE 15
#define MAXCHAR 150

//pthread parameter struct
typedef struct ptparam {
	sem_t mutex, full, empty;
	char buff[BUFFSIZE];
	FILE *fp;
} ptparam;

//functions
void *producer(void *p);
void *consumer(void *p);

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

	//sets file pointer into shared mem & sets flag
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
void *producer(void *p) {

	int i = 0;
	char c;
	ptparam *buff = (ptparam*)p;
	while(i < MAXCHAR && (c = getc(buff->fp)) != EOF) {

		//waits till empty and mutex are positive and decrements
		sem_wait(&buff->empty);
		sem_wait(&buff->mutex);

		//writes to buffer
		buff->buff[i % BUFFSIZE] = c;
		i++;

		//increments mutex and full
		sem_post(&buff->mutex);
		sem_post(&buff->full);
	}

	//writes the final * stopping char into buffer
	sem_wait(&buff->empty);
	sem_wait(&buff->mutex);

	buff->buff[i % BUFFSIZE] = '*';

	sem_post(&buff->mutex);
	sem_post(&buff->full);
}

//consumer function
void *consumer(void *p) {

	int i = 0;
	char c = '0';
	ptparam *buff = (ptparam*)p;
	while(c != '*') {

		//waits till full and mutex are positive and decrements
		sem_wait(&buff->full);
		sem_wait(&buff->mutex);

		//reads from the buffer and prints unless its *
		c = buff->buff[i % BUFFSIZE];
		i++;
		if(c != '*')
			printf("%c", c);

		//increments mutex and empty
		sem_post(&buff->mutex);
		sem_post(&buff->empty);
	}
}