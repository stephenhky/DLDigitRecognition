package home.dl4j;

import home.dl4j.dao.MNISTFactory;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

/**
 * Created by hok on 1/19/17.
 */
public class DenseDLRecognitionTesting {

    public static void main(String[] args) throws IOException {
        // get training data
        DataSetIterator mnistTrain = MNISTFactory.getMNISTTrainDataIterator();

        // initialize neural network
        MultiLayerNetwork nnet = NeuralNets.RetrieveSingleDenseLayer(MNISTFactory.numColumns*MNISTFactory.numRows,
                MNISTFactory.outputNum,
                MNISTFactory.rngSeed);
        nnet.init();

        // training
        for (int i=0; i<MNISTFactory.numEpochs; i++) {
            nnet.fit(mnistTrain);
        }

        // evaluation
        DataSetIterator mnistTest = MNISTFactory.getMNISTTestDataIterator();
        Evaluation evaluation = new Evaluation(MNISTFactory.outputNum);
        while (mnistTest.hasNext()) {
            DataSet dataSet = mnistTest.next();
            INDArray prediction = nnet.output(dataSet.getFeatureMatrix());
            evaluation.eval(dataSet.getLabels(), prediction);
            System.out.println("label = "+dataSet.getLabels()+" / prediction = "+prediction);
        }
        System.out.println(evaluation.stats());
    }
}
