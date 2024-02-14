#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "parameters.h"

void writeResult_matrix(double **image_out, const char* resultado_filename, int numberOfSamples, int numberOfLines, int numberOfBands);
void writeResult_char(char *image_out, const char* resultado_filename, int numberOfSamples, int numberOfLines, int numberOfBands);
void writeHeader(const char* outHeader, int numberOfSamples, int numberOfLines, int numberOfBands);

//uint16_t *readRawImageUint16(char *hsiCube, int	numberOfLines, int numberOfSamples, int	numberOfBands);
void readNormalizedImage(char *image, int numberOfPixels, int	numberOfBands, float* normalizedImage);
//float *rawImageCalibrationUint16(uint16_t *rawImage, int numberOfLines, int numberOfSamples, int numberOfBands);
float avg(float *score, int initialValue, int size);
void mapMinMax(float *vectorIn, float *vectorOut, int yMin, int yMax, int size);

/*PCA functions*/
double **pcaOneBand( FILE *f_time, int numberOfLines, int numberOfSamples, int numberOfBands, int numberOfPcaBands, float *image_in);
/*SVM functions*/
float **svmPrediction(FILE *f_time, int	numberOfLines, int	numberOfSamples, int	numberOfBands, int	numberOfClasses, float	*image_in);
/*KNN functions*/
char *knnFiltering(FILE *f_time, int numberOfLines, int	numberOfSamples, int	numberOfBands, int	numberOfClasses, double	**pcaOneBandResult, float	**svmProbabilityMapResult);