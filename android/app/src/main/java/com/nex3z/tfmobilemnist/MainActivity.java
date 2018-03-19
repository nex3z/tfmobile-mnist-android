package com.nex3z.tfmobilemnist;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import com.nex3z.fingerpaintview.FingerPaintView;

import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;

public class MainActivity extends AppCompatActivity {
    private static final String LOG_TAG = MainActivity.class.getSimpleName();

    @BindView(R.id.fpv_paint) FingerPaintView mFpvPaint;
    @BindView(R.id.tv_prediction) TextView mTvPrediction;
    @BindView(R.id.tv_probability) TextView mTvProbability;
    @BindView(R.id.tv_timecost) TextView mTvTimeCost;

    private Classifier mClassifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ButterKnife.bind(this);
        init();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mClassifier.close();
    }

    @OnClick(R.id.btn_detect)
    void onDetectClick() {
        if (mClassifier == null) {
            Log.e(LOG_TAG, "onDetectClick(): Classifier is not initialized");
            return;
        } else if (mFpvPaint.isEmpty()) {
            Toast.makeText(this, R.string.please_write_a_digit, Toast.LENGTH_SHORT).show();
            return;
        }

        Bitmap image = mFpvPaint.exportToBitmap(
                Classifier.DIM_IMG_SIZE_WIDTH, Classifier.DIM_IMG_SIZE_HEIGHT);
        Bitmap inverted = ImageUtil.invert(image);
        Result result = mClassifier.classify(inverted);
        renderResult(result);
    }

    @OnClick(R.id.btn_clear)
    void onClearClick() {
        mFpvPaint.clear();
        mTvPrediction.setText(R.string.empty);
        mTvProbability.setText(R.string.empty);
        mTvTimeCost.setText(R.string.empty);
    }

    private void init() {
        try {
            mClassifier = new Classifier(this);
        } catch (RuntimeException e) {
            Log.e(LOG_TAG, "Failed to create classifier.", e);
            Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void renderResult(Result result) {
        mTvPrediction.setText(String.valueOf(result.getNumber()));
        mTvProbability.setText(String.valueOf(result.getProbability()));
        mTvTimeCost.setText(String.format(getString(R.string.timecost_value),
                result.getTimeCost()));
    }

}
