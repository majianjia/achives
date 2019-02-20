
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "rtthread.h"

#include "nnom.h"
#include "m_nnom.h"

#include "weights.h"

#include "data_structures.h"
#include "ymodem.h"


// NN rate
#define NN_RATE				20 //HZ

#define INPUT_RATE			50
#define INPUT_CH			3	 
#define INPUT_WIDTH			128
#define INPUT_HIGHT			1
#define DATA_TYPE_COUNT  	(4)

#define FILTER_SIZE 2
float avg_filter[FILTER_SIZE][3] = {0};
int8_t filter_index = 0;

// 
static struct rt_semaphore sem;
static rt_timer_t timer;

uint32_t start_time, stop_time, nn_time_mes;
struct _nn_time {
	uint32_t max;
	uint32_t avg;
	struct {
		int32_t max_last;
		uint32_t avg_count;
		uint32_t avg_accu;
	}p;
} nn_time;


static struct _nn_buf {
	bool initialized;
	rt_mutex_t mutex;		// mutex for accessing buffer
	uint32_t index;
	uint32_t data_size;
	uint32_t buf_size;
	uint8_t * buf;
} nn_data_buf = {.initialized = false,};

static float limit(float input, float min, float max)
{
	if(input < min) input = min;
	if(input > max) input = max;
	return input;
}

static void add_data(struct _nn_buf *buf, int8_t * data)
{	
	if(buf->initialized!=true)
		return;
	rt_mutex_take(buf->mutex, RT_WAITING_FOREVER);
	for(uint32_t i = 0; i < buf->data_size; i++)
	{
		buf->buf[buf->index * buf->data_size +i] = data[i];
	}
	buf->index +=1;
	if(buf->index >= buf->buf_size)
		buf->index = 0;
	rt_mutex_release(buf->mutex);
}
static void get_data(struct _nn_buf *buf, int8_t * data)
{
	if(buf->initialized!=true)
		return;
	rt_mutex_take(buf->mutex, RT_WAITING_FOREVER);
	if(buf->index == 0){
		rt_memcpy(data, buf->buf, buf->buf_size * buf->data_size);
	}
	else{
		uint32_t index = buf->index * buf->data_size;
		uint32_t size = buf->buf_size * buf->data_size;
		
		rt_memcpy(data, buf->buf + index, size - index);
		rt_memcpy(data + (size - index), buf->buf, index);
	}
	rt_mutex_release(buf->mutex);
}

static void filter_add(float a, float b,float c)
{
	avg_filter[filter_index][0] = a;
	avg_filter[filter_index][1] = b;
	avg_filter[filter_index][2] = c;
	filter_index++;
	if(filter_index >= FILTER_SIZE)
		filter_index = 0;
}
static void filter_get(float *a, float *b, float *c)
{
	for(uint32_t i = 0; i < FILTER_SIZE; i++)
	{
		*a += avg_filter[i][0];
		*b += avg_filter[i][1];
		*c += avg_filter[i][2];
	}
	*a /= FILTER_SIZE;
	*b /= FILTER_SIZE;
	*c /= FILTER_SIZE;
}

void thread_nn_data(void *p)
{
	float acc, rotate, a, b, c;
	uint32_t rate = INPUT_RATE;
	int8_t data[INPUT_CH];

	// init buffer
	nn_data_buf.buf_size = INPUT_WIDTH;
	nn_data_buf.data_size = INPUT_CH;
	nn_data_buf.buf = rt_malloc(INPUT_WIDTH * INPUT_CH);
	nn_data_buf.initialized = true;
	
	while(1)
	{
		// if low power, stop the timer
		if(system_status.power_mode != SYSTEM_FULL_POWER)
		{
			rt_thread_delay(1000);
			continue;
		}
		// if not low power ,start timer to get the nn frequency. 
		else
		{
			rt_thread_delay(1000/rate);
		}
		
		// get data
		a = motion_data.s_acc;
		b = motion_data.s_rotate;
		c = loadcell.force;
		
		// filter them
		filter_add(a, b, c);
		filter_get(&a, &b, &c);
		
		data[0] = limit(a * 4, -128, 127);
		data[1] = limit(b / 8, -128, 127);
		data[2] = limit(c * 128.f / 100.f, 0, 127); 
		
		
		add_data(&nn_data_buf, data);
		
	}
}

// timer for run frequency
static void timeout(void * parameter)
{
	rt_sem_release(&sem);
}
static void time_mea_start()
{
	start_time = capture_current_timer();
}
static void time_mea_calc()
{
	stop_time = capture_current_timer();
	nn_time_mes =  stop_time - start_time;
	// calculate timing
	// max
	nn_time.p.max_last --;
	if(nn_time.p.max_last <=0 || nn_time_mes > nn_time.max)
	{
		nn_time.max = nn_time_mes;
		nn_time.p.max_last = NN_RATE * 3;
	}
	nn_time.p.max_last --;
	// avg
	nn_time.p.avg_accu += nn_time_mes;
	nn_time.p.avg_count ++;
	if(nn_time.p.avg_count > NN_RATE*3)
	{
		nn_time.avg = nn_time.p.avg_accu / nn_time.p.avg_count;
		nn_time.p.avg_count = 0;
		nn_time.p.avg_accu = 0;
	}
}
	


// NN ----------------
const int8_t conv1_wt[] = CONV1D_1_KERNEL_0;
const int8_t conv1_b[] = CONV1D_1_BIAS_0;
const int8_t conv2_wt[] = CONV1D_2_KERNEL_0;
const int8_t conv2_b[] = CONV1D_2_BIAS_0;
const int8_t conv3_wt[] = CONV1D_3_KERNEL_0;
const int8_t conv3_b[] = CONV1D_3_BIAS_0;
const int8_t conv4_wt[] = CONV1D_4_KERNEL_0;
const int8_t conv4_b[] = CONV1D_4_BIAS_0;
const int8_t conv5_wt[] = CONV1D_5_KERNEL_0;
const int8_t conv5_b[] = CONV1D_5_BIAS_0;
const int8_t fc1_wt[] = DENSE_1_KERNEL_0;
const int8_t fc1_b[] = DENSE_1_BIAS_0;
const int8_t fc2_wt[] = DENSE_2_KERNEL_0;
const int8_t fc2_b[] = DENSE_2_BIAS_0;

nnom_weight_t c1_w = {
	.p_value = (void*)conv1_wt,
	.shift = CONV1D_1_KERNEL_0_SHIFT};

nnom_bias_t c1_b = {
	.p_value = (void*)conv1_b,
	.shift = CONV1D_1_BIAS_0_SHIFT};

nnom_weight_t c2_w = {
	.p_value = (void*)conv2_wt,
	.shift = CONV1D_2_KERNEL_0_SHIFT};

nnom_bias_t c2_b = {
	.p_value = (void*)conv2_b,
	.shift = CONV1D_2_BIAS_0_SHIFT};

nnom_weight_t c3_w = {
	.p_value = (void*)conv3_wt,
	.shift = CONV1D_3_KERNEL_0_SHIFT};

nnom_bias_t c3_b = {
	.p_value = (void*)conv3_b,
	.shift = CONV1D_3_BIAS_0_SHIFT};

nnom_weight_t c4_w = {
	.p_value = (void*)conv4_wt,
	.shift = CONV1D_4_KERNEL_0_SHIFT};

nnom_bias_t c4_b = {
	.p_value = (void*)conv4_b,
	.shift = CONV1D_4_BIAS_0_SHIFT};

nnom_weight_t c5_w = {
	.p_value = (void*)conv5_wt,
	.shift = CONV1D_5_KERNEL_0_SHIFT};

nnom_bias_t c5_b = {
	.p_value = (void*)conv5_b,
	.shift = CONV1D_5_BIAS_0_SHIFT};

nnom_weight_t ip1_w = {
	.p_value = (void*)fc1_wt,
	.shift = DENSE_1_KERNEL_0_SHIFT};

nnom_bias_t ip1_b = {
	.p_value = (void*)fc1_b,
	.shift = DENSE_1_BIAS_0_SHIFT};

nnom_weight_t ip2_w = {
	.p_value = (void*)fc2_wt,
	.shift = DENSE_2_KERNEL_0_SHIFT};

nnom_bias_t ip2_b = {
	.p_value = (void*)fc2_b,
	.shift = DENSE_2_BIAS_0_SHIFT};

	
nnom_model_t model = {0}; // to use finsh to print
int8_t nnom_input_data[INPUT_WIDTH * INPUT_CH];
int8_t nnom_output_data[6];


void thread_nnom(void *p)
{
	nnom_layer_t *input_layer;
	nnom_layer_t *x;
	nnom_layer_t *x1;
	nnom_layer_t *x2;
	nnom_layer_t *x3;
	void nnom_get_data(void * data);
	
	rt_thread_delay(1000);

	new_model(&model);
	
	// input format
	input_layer = Input(shape(1, 128, 3), qformat(7, 0), nnom_input_data);
	
	// conv2d
	x = model.hook(Conv2D(8, kernel(1, 11), stride(1, 2), PADDING_SAME, &c1_w, &c1_b), input_layer);
	x = model.active(act_relu(), x);
	x = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x);
	
	// conv2d - 1 - inception
	x1 = model.hook(Conv2D(8, kernel(1, 7), stride(1, 1), PADDING_SAME, &c2_w, &c2_b), x);
	x1 = model.active(act_relu(), x1);
	x1 = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x1);
	
	// conv2d - 2 - inception
	x2 = model.hook(Conv2D(8, kernel(1, 3), stride(1, 1), PADDING_SAME, &c3_w, &c3_b), x);
	x2 = model.active(act_relu(), x2);
	x2 = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x2);
	
	// maxpool - 3 - inception
	x3 = model.hook(Conv2D(8, kernel(1, 1), stride(1, 1), PADDING_SAME, &c4_w, &c4_b), x);
	x3 = model.active(act_relu(), x3);
	x3 = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x3);
	
	// concatenate 
	x = model.merge(Concat(-1), x1, x2); 
	x = model.merge(Concat(-1), x, x3);
	
	// conv2d conclusion of inception 
	x = model.hook(Conv2D(24, kernel(1, 3), stride(1, 1), PADDING_SAME, &c5_w, &c5_b), x);
	x = model.active(act_relu(), x);
	x = model.hook(MaxPool(kernel(1, 2), stride(1, 2), PADDING_VALID), x);
	
	// flatten & dense
	x = model.hook(Flatten(), x);
	x = model.hook(Dense(64, &ip1_w, &ip1_b), x);
	x = model.active(act_relu(), x);
	x = model.hook(Dense(DATA_TYPE_COUNT, &ip2_w, &ip2_b), x);
	x = model.hook(Softmax(), x);
	x = model.hook(Output(shape(DATA_TYPE_COUNT,1,1), qformat(7, 0), nnom_output_data), x);
	
	// compile and check
	model_compile(&model, input_layer, x);
	
	while(1)
	{
		rt_sem_take(&sem, 1000);
		// if low power, stop the timer
		if(system_status.power_mode != SYSTEM_FULL_POWER)
		{
			rt_timer_stop(timer);
			continue;
		}
		// if not low power ,start timer to get the nn frequency. 
		else
		{
			rt_timer_start(timer);
		}
		
		// NOW RUN !!!
		time_mea_start();
		get_data(&nn_data_buf, nnom_input_data);
		model_run(&model);
		time_mea_calc();
	}
}

void thread_nnom_init()
{
	rt_thread_t tid;
	
	//init the semaphore for 
	if (rt_object_find("nn_data", RT_Object_Class_Semaphore) == RT_NULL)
		rt_sem_init(&sem, "nn_data", 0, RT_IPC_FLAG_FIFO);
	
	// init data buf mutex
	nn_data_buf.mutex = rt_mutex_create("nn_data", RT_IPC_FLAG_FIFO);
	
	timer = rt_timer_create("nn",timeout, RT_NULL, (int)(RT_TICK_PER_SECOND/NN_RATE), RT_TIMER_FLAG_PERIODIC);
	rt_timer_start(timer);

	tid = rt_thread_create("nnom", thread_nnom, RT_NULL, 1024, 30, 500);
	rt_thread_startup(tid);
	
	tid = rt_thread_create("nn_data", thread_nn_data, RT_NULL, 512, 29, 500);
	rt_thread_startup(tid);
}





#ifdef RT_USING_FINSH
#include <finsh.h>
#include "math.h"
void nn_stat()
{
	void cpu_usage_get(rt_uint8_t *major, rt_uint8_t *minor);
	unsigned char major, minor;
	model_stat(&model);
	rt_kprintf("NNOM: Max run time: %d us\n", nn_time.max);
	rt_kprintf("NNOM: Mean run time: %d us\n", nn_time.avg);
	rt_kprintf("NNOM: Total Mem: %d\n", nnom_mem_stat());
	(void) cpu_usage_get(&major, &minor);
	rt_kprintf("CPU usage: %d.%02d%% \n", major, minor);
}

FINSH_FUNCTION_EXPORT(nn_stat, nn_stat() to print data);

#endif

// test -------------------------- Using Y-modem to send test data set. 

#ifdef RT_USING_FINSH
#include <finsh.h>
#include "rtdevice.h"

#define DATA_SIZE (INPUT_CH * INPUT_WIDTH * INPUT_HIGHT)
#define LABEL_SIZE 128

static size_t file_total_size, file_cur_size;

//test
struct rt_ringbuffer  ringbuffer;
int32_t rb_size;
uint8_t rb_pool[2048]; 

nnom_predic_t *prediction = NULL;

// parameters
uint8_t	 test_label[LABEL_SIZE] = {0};  // where a batch of label stores
uint32_t test_label_countdown = 0;		// count down of that batch
uint32_t test_total_count = 0;			// 

static enum rym_code ymodem_on_begin(struct rym_ctx *ctx, rt_uint8_t *buf, rt_size_t len) {
	char *file_name, *file_size;

	/* calculate and store file size */
	file_name = (char *) &buf[0];
	file_size = (char *) &buf[rt_strlen(file_name) + 1];
	file_total_size = atol(file_size);
	/* 4 bytes align */
	file_total_size = (file_total_size + 3) / 4 * 4;
	file_cur_size = 0;
	
	// local data size
	test_label_countdown = 0;
	test_total_count = 0;
	memset(test_label, 0, LABEL_SIZE);
	
	return RYM_CODE_ACK;
}

static enum rym_code ymodem_on_data(struct rym_ctx *ctx, rt_uint8_t *buf, rt_size_t len) 
{
	// put data in buffer, then get it as block. 
	rt_ringbuffer_put(&ringbuffer, buf, len);
	rb_size += len;
	
	while(1)
	{
		// get label. 
		if(test_label_countdown == 0 && rb_size >= LABEL_SIZE)
		{
			// get the label, reset the label countdown. 	
			rt_ringbuffer_get(&ringbuffer, &test_label[0], LABEL_SIZE);
			rb_size -= LABEL_SIZE;
			
			test_label_countdown = LABEL_SIZE;
		}
		
		// if there is enough data and the label is still availble. 
		if(test_label_countdown > 0 && rb_size >= DATA_SIZE)
		{
			// use one lata
			test_label_countdown --;
			
			// get input data
			rt_ringbuffer_get(&ringbuffer, &nnom_input_data[0], DATA_SIZE);
			rb_size -= DATA_SIZE;
			
			// do this prediction round.
			prediction_run(prediction, test_label[test_total_count % LABEL_SIZE]);
			
			// we can use the count in prediction as well.
			test_total_count += 1;
		}
		// return while there isnt enough data
		else
		{
			return RYM_CODE_ACK;
		}
	}
}



extern void sync_timer_start();
extern rt_err_t sync_timer_stop();


void predic() 
{
	struct rym_ctx rctx;

	rt_kprintf("Please select the NNoM binary test file and use Ymodem-128/1024  to send.\n");

	// preparing for prediction 
	sync_timer_start();
	
	rt_ringbuffer_init(&ringbuffer, rb_pool, 2048);
	rb_size = 0;
	
	// delete first if it its not freed
	if(prediction!=NULL)
		predicetion_delete(prediction);
	
	// create new instance (test with all k)
	prediction = prediction_create(&model, nnom_output_data, DATA_TYPE_COUNT, DATA_TYPE_COUNT-1);
	
	// begin
	// data is feed in receiving callback
	if (!rym_recv_on_device(&rctx, rt_console_get_device(), RT_DEVICE_OFLAG_RDWR | RT_DEVICE_FLAG_INT_RX,
			ymodem_on_begin, ymodem_on_data, NULL, RT_TICK_PER_SECOND)) {
		/* wait some time for terminal response finish */
		rt_thread_delay(RT_TICK_PER_SECOND / 10);
		rt_kprintf("\nPrediction done.\n");

	} else {
		/* wait some time for terminal response finish */
		rt_thread_delay(RT_TICK_PER_SECOND / 10);
		rt_kprintf("Test file incompleted. \n Partial results are shown below:\n");
	}
	// finished
	prediction_end(prediction);
	// print sumarry & matrix
	prediction_summary(prediction);
	
	sync_timer_stop();

}
FINSH_FUNCTION_EXPORT(predic, validate NNoM model implementation with test set);
MSH_CMD_EXPORT(predic, validate NNoM model implementation with test set);

void matrix()
{
	if(prediction != NULL)
		prediction_matrix(prediction);
}
FINSH_FUNCTION_EXPORT(matrix, matrix() to print confusion matrix);
MSH_CMD_EXPORT(matrix, print confusion matrix);

#endif

// support for printf
#ifdef __MICROLIB
#include <stdio.h>

int fputc(int c, FILE *f) 
{
    char ch[2] = {0};
    ch[0] = c;
    rt_kprintf(&ch[0]);
    return 1;
}

int fgetc(FILE *f) 
{
#ifdef RT_USING_POSIX
    char ch;
    if (libc_stdio_read(&ch, 1) == 1)
        return ch;
#endif
    return -1;
}
#endif










