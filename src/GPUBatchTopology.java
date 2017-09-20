/**
 * Created by hadoop on 4/28/17.
 */
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Map;
import java.util.Random;

import backtype.storm.metric.api.GlobalMetrics;
import com.twitter.heron.api.HeronSubmitter;
import com.twitter.heron.api.spout.BaseRichSpout;
import com.twitter.heron.api.spout.SpoutOutputCollector;
import com.twitter.heron.api.topology.OutputFieldsDeclarer;
import com.twitter.heron.api.topology.TopologyContext;
import com.twitter.heron.api.tuple.Fields;
import com.twitter.heron.api.tuple.Values;

import com.twitter.heron.api.bolt.BaseBatchBolt;
import com.twitter.heron.api.topology.TopologyBuilder;
import com.twitter.heron.api.bolt.OutputCollector;
import com.twitter.heron.api.tuple.Tuple;
import com.twitter.heron.api.Config;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_1D;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import jcuda.jcufft.JCufft;
import jcuda.jcufft.cufftHandle;
import jcuda.jcufft.cufftType;

public class GPUBatchTopology {
    private static int matrix_size = 60;
    private static boolean isGPU = true;
    public GPUBatchTopology() {
    }

    public static void main(String[] args) throws Exception {
        TopologyBuilder builder = new TopologyBuilder();
        int parallelism = 1;
        /* we define two parallel spouts and 4 bolts here */
        builder.setSpout("matrix", new GPUBatchTopology.GPUSpout(), parallelism);
        builder.setBolt("testBolt", new GPUBatchTopology.GPUBatchBolt(), 2 * parallelism).shuffleGrouping("matrix");
        builder.setBolt("testBolt1", new GPUBatchTopology.FFTBatchBolt(), 1 * parallelism).shuffleGrouping("testBolt");

        Config conf = new Config();
        conf.setDebug(true);


        if (args != null && args.length > 0) {
            conf.setNumStmgrs(parallelism);
            HeronSubmitter.submitTopology(args[0], conf, builder.createTopology());
        }
    }

    private static float[] createRandomFloatData(int n)
    {
        Random random = new Random();
        float x[] = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = random.nextFloat();
        }
        return x;
    }

    public static class GPUSpout extends BaseRichSpout{
        /*
         * generate random matrixes with size n * n
        */
        private static final long serialVersionUID = 4322775001819135036L;

        private static final int n = matrix_size;
        private static final int matSeqLen = 20;
        private  ArrayList<float[]> list;

        private final Random rnd = new Random(31);

        private SpoutOutputCollector collector;

        public void declareOutputFields(OutputFieldsDeclarer outputFieldsDeclarer) {
            outputFieldsDeclarer.declare(new Fields("matrix"));
        }

        @SuppressWarnings("rawtypes")
        public void open(Map map, TopologyContext topologyContext,
                         SpoutOutputCollector spoutOutputCollector) {
            list = new ArrayList<>();
            for (int i = 0; i < matSeqLen; i++) {
                list.add(createRandomFloatData(n * n));
            }

            collector = spoutOutputCollector;
        }

        @Override
        public void nextTuple() {
            int nextInt = rnd.nextInt(matSeqLen);
            /* only bytes are availiable*/
            collector.emit(new Values( convertf2b(list.get(nextInt), n * n)));
        }

        /* convert float array to a byte array */
        private byte[] convertf2b(float [] data, int size){
            ByteBuffer byteBuffer = ByteBuffer.allocate(4 * size);
            FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
            floatBuffer.put(data);
            byte [] res = byteBuffer.array();
            return res;
        }
    }

    public static class FFTBatchBolt extends BaseBatchBolt {
        private long nItems;
        private long startTime;
        private OutputCollector collector;
        private int length = matrix_size * matrix_size;

        @SuppressWarnings("rawtypes")
        public void prepare(Map conf, TopologyContext contect, OutputCollector collector) {
            nItems = 0;
            startTime = System.currentTimeMillis();
            this.collector = collector;
        }

        @Override
        public void execute(ArrayList<Tuple> tupleList) {
            int size = tupleList.size();
            Tuple t;
            byte [] buffer;
            float [] input;
            float [] intermidOutPut;
            if (isGPU) {
                cufftHandle plan = new cufftHandle();
                for (int i = 0; i < size; i++) {
                    t = tupleList.get(i);
                    buffer = t.getBinary(0);
                    input = convertb2f(buffer);
                    intermidOutPut = input.clone();
                    JCufft.cufftPlan1d(plan, length / 2, cufftType.CUFFT_C2C, 1);
                    JCufft.cufftExecC2C(plan, intermidOutPut, intermidOutPut, JCufft.CUFFT_FORWARD);
                    GlobalMetrics.incr("selected_items");
                    collector.emit(t, new Values(convertf2b(intermidOutPut, length)));
                    collector.ack(t);
                    nItems+=1;
                    long latency = System.currentTimeMillis() - startTime;
                    if (nItems == 50000) {
                        System.out.println("Bolt processed " + nItems + " tuples in " + latency + " ms");
                    }
                }
                JCufft.cufftDestroy(plan);
//                long latency = System.currentTimeMillis() - startTime;
//                nItems += size;
//                System.out.println("Bolt processed " + nItems + " batches in " + latency + " ms");
            } else {
                for (int i = 0; i < size; i++) {
                    t = tupleList.get(i);
                    buffer = t.getBinary(0);
                    input = convertb2f(buffer);
                    intermidOutPut = input.clone();
                    FloatFFT_1D fft = new FloatFFT_1D(length / 2);
                    fft.complexForward(intermidOutPut);
                    GlobalMetrics.incr("selected_items");
                    collector.emit(t, new Values(convertf2b(intermidOutPut, length)));
                    collector.ack(t);
                    nItems+=1;
                    long latency = System.currentTimeMillis() - startTime;
                    if (nItems == 50000) {
                        System.out.println("Bolt processed " + nItems + " tuples in " + latency + " ms");
                    }
                }
//                long latency = System.currentTimeMillis() - startTime;
//                nItems += size;
//                System.out.println("Bolt processed " + nItems + " tuples in " + latency + " ms");
            }
        }

        public void declareOutputFields(OutputFieldsDeclarer outputFieldsDeclarer) {
            outputFieldsDeclarer.declare(new Fields("out3")); // this must be specified, otherwise the meited result can't be used as an input stream
        }

        /* convert bytes to float[]*/
        private float[] convertb2f(byte [] buffer){
            ByteBuffer bbf = ByteBuffer.wrap(buffer);
            FloatBuffer fb = bbf.asFloatBuffer();
            float [] floatArray = new float[fb.limit()];
            fb.get(floatArray);
            return floatArray;
        }

        private byte[] convertf2b(float [] data, int size){
            ByteBuffer byteBuffer = ByteBuffer.allocate(4 * size);
            FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
            floatBuffer.put(data);
            byte [] res = byteBuffer.array();
            return res;
        }
    }


    public static class GPUBatchBolt extends BaseBatchBolt {
        private static final int n = matrix_size;
        private static final int nn = matrix_size * matrix_size;
        private static final boolean isGPU = GPUBatchTopology.isGPU;
        private long nItems;
        private long startTime;
        private float[] matrix1 = createRandomFloatData(nn);
        private float[] matrix2 = createRandomFloatData(nn);
        private float alpha = 0.3f;
        private float beta = 0.7f;
        private Pointer p1, p2, p3;
        private OutputCollector collector;

        @SuppressWarnings("rawtypes")
        public void prepare(Map conf, TopologyContext contect, OutputCollector collector) {
            nItems = 0;
            startTime = System.currentTimeMillis();
            this.collector = collector;
        }

        @Override
        public void execute(ArrayList<Tuple> tupleList) {
            int size = tupleList.size();
            Tuple t;
            byte [] buffer;
            float [] input;
            //System.out.println("start batch processing with isGPU = " + isGPU + " batch size = " + size);
            JCublas.cublasInit(); // init
            p1 = new Pointer();
            p2 = new Pointer();
            p3 = new Pointer();
            // allocate memory in GPU
            if (isGPU) {
                JCublas.cublasAlloc(nn, Sizeof.FLOAT, p1);
                JCublas.cublasAlloc(nn, Sizeof.FLOAT, p2);
                JCublas.cublasAlloc(nn, Sizeof.FLOAT, p3);
            }
            for (int i = 0; i < size; i++) {
                t = tupleList.get(i);
                buffer = t.getBinary(0);
                input = convertb2f(buffer);
                if (!isGPU) {
                    sgemmJava(n, alpha, input, matrix1, beta, matrix2);
                }
                else {
                    JCublas.cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(input), 1, p1, 1);
                    JCublas.cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(matrix1), 1, p2, 1);
                    JCublas.cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(matrix2), 1, p3, 1);
                    // Execute sgemm
                    JCublas.cublasSgemm('n', 'n', n, n, n, alpha, p1, n, p2, n, beta, p3, n);
                    JCublas.cublasGetVector(nn, Sizeof.FLOAT, p3, 1, Pointer.to(matrix2), 1);
                }
                GlobalMetrics.incr("selected_items");
                collector.emit(t, new Values( convertf2b(matrix2, n * n)));
                //System.out.println("component=>"+compId+"-"+taskId+" - word=>"+word + ", " + "count=>"+val);
                collector.ack(t);
            }
            if (isGPU) {
                JCublas.cublasFree(p1);
                JCublas.cublasFree(p2);
                JCublas.cublasFree(p3);
                JCublas.cublasShutdown();
                //System.out.println("cuda resources are cleaned up");
            }

            long latency = System.currentTimeMillis() - startTime;
            nItems+=size;
            System.out.println("Bolt processed " + nItems + " batches in " + latency + " ms");
        }

        @Override
        public void cleanup(){
            //System.out.println("cuda resources are cleaned up");
        }

        public void declareOutputFields(OutputFieldsDeclarer outputFieldsDeclarer) {
            outputFieldsDeclarer.declare(new Fields("out2")); // this must be specified, otherwise the meited result can't be used as an input stream
        }

        private static void sgemmJava(int n, float alpha, float A[], float B[],
                                      float beta, float C[])
        {
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    float prod = 0;
                    for (int k = 0; k < n; ++k)
                    {
                        prod += A[k * n + i] * B[j * n + k];
                    }
                    C[j * n + i] = alpha * prod + beta * C[j * n + i];
                }
            }
        }

        /* convert bytes to float[]*/
        private float[] convertb2f(byte [] buffer){
            ByteBuffer bbf = ByteBuffer.wrap(buffer);
            FloatBuffer fb = bbf.asFloatBuffer();
            float [] floatArray = new float[fb.limit()];
            fb.get(floatArray);
            return floatArray;
        }

        private byte[] convertf2b(float [] data, int size){
            ByteBuffer byteBuffer = ByteBuffer.allocate(4 * size);
            FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
            floatBuffer.put(data);
            byte [] res = byteBuffer.array();
            return res;
        }
    }
}
