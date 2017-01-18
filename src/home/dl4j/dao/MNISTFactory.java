package home.dl4j.dao;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

/**
 * Created by hok on 1/18/17.
 */
public class MNISTFactory {
    public static final int numRows = 28;
    public static final int numColumns = 28;
    public static int outputNum = 10;
    public static int batchSize = 128;
    protected static int rngSeed = 123;
    protected static int numEpochs = 15;

    public static DataSetIterator getMNISTTrainDataIterator() throws IOException {
        return new MnistDataSetIterator(batchSize, true, rngSeed);
    }

    public static DataSetIterator getMNISTTestDataIterator() throws IOException {
        return new MnistDataSetIterator(batchSize, false, rngSeed);
    }
}
