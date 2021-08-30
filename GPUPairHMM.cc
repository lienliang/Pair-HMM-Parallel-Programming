/*
    Author: Enliang Li
    Main Entry point for deliverables supporting IntelPairHmm
    Latest Version: 1.0 on Feb.27th 2020

    *********************
    Definition for "testcase"
    typedef struct {
        size_t rs_len, hap_len;
        vector <uint8_t> q;
        vector <uint8_t> i;
        vector <uint8_t> d;
        vector <uint8_t> c;
        vector <uint8_t> hap;
        vector <uint8_t> rs;
    } testcase;
    *********************
    q >> readQuals
    i >> insrtionGOP
    d >> deletionGOP
    c >> overallGCP
    hap >> haplotypeBases
    rs >> readBases

*/

#include <vector>
#include <ctime>
#include <math.h>
#include <assert.h>
#include <iostream>

#include "forward.cuh"
#include "input_reader.h"
#include "testcase.h"
#include "testcase_iterator.h"

using namespace std;

double *g_compute_full_prob_double_gpu(std::vector<testcase> tc) {

    int num_batch = tc.size();

    //Where all results stored (dynamically allocated)
    double * result_batch = new double[num_batch]();

    //Where only cur round of calculation results
    double *cur_batch = new double[BATCH_REG];
    int effective_copy = BATCH_REG;

    unsigned int *effective_rslen = new unsigned int [BATCH_REG]();
    unsigned int *effective_haplen = new unsigned int [BATCH_REG]();

    unsigned char *haplotypeBases = new unsigned char[BATCH_REG * HAPLEN]();
    unsigned char *readBases = new unsigned char[BATCH_REG * RSLEN]();
    unsigned char *readQuals = new unsigned char[BATCH_REG * RSLEN]();
    unsigned char *insertionGOP = new unsigned char[BATCH_REG * RSLEN]();
    unsigned char *deletionGOP = new unsigned char[BATCH_REG * RSLEN]();
    unsigned char *overallGCP = new unsigned char[BATCH_REG * RSLEN]();

    int no_of_batch_result = 0;

    //Input data

    while (num_batch > 0) {
        cout << " num of batch in current stream: " << num_batch << endl << endl;
        if (num_batch < BATCH_REG) {
            effective_copy = num_batch;
        }
        else {
            effective_copy = BATCH_REG;
        }

        for (int batch_id = 0; batch_id < effective_copy; batch_id++) {

            unsigned int cur_copy_effective_rslen = tc[batch_id].rs_len;
            unsigned int cur_copy_effective_haplen = tc[batch_id].hap_len;


            effective_rslen[batch_id] = cur_copy_effective_rslen;
            effective_haplen[batch_id] = cur_copy_effective_haplen;

            for (int read = 0; read <  cur_copy_effective_rslen; read++){
                readBases[batch_id * RSLEN + read] = (unsigned char) (tc[batch_id]).rs[read];
                readQuals[batch_id * RSLEN + read] = (unsigned char) (tc[batch_id]).q[read];
                insertionGOP[batch_id * RSLEN + read] = (unsigned char) (tc[batch_id]).i[read];
                deletionGOP[batch_id * RSLEN + read] = (unsigned char) (tc[batch_id]).d[read];
                overallGCP[batch_id * RSLEN + read] = (unsigned char) (tc[batch_id]).c[read];
            }

            for (int ref = 0; ref < cur_copy_effective_haplen; ref++) {
                haplotypeBases[batch_id * HAPLEN + ref] = (unsigned char) (tc[batch_id]).hap[ref];
            }

        }

        //call the appropriate function API
        subComputeReadLikelihoodGivenHaplotypeLog10(
            cur_batch,
            effective_copy,
            effective_rslen,
            effective_haplen,
            haplotypeBases,
            readBases,
            readQuals,
            insertionGOP,
            deletionGOP,
            overallGCP
        );

        for (int batch_id = 0; batch_id < effective_copy; batch_id++) {
            result_batch[no_of_batch_result] = cur_batch[batch_id];
            no_of_batch_result++;
        }

        
        num_batch -= BATCH_REG;
    }

    // free all self-allocated item
    delete(cur_batch);
    delete(effective_rslen);
    delete(effective_haplen);
    delete(haplotypeBases);
    delete(readBases);
    delete(readQuals);
    delete(insertionGOP);
    delete(deletionGOP);
    delete(overallGCP);

    return result_batch;
}


/*

    *********************
    Definition for "testcase"
    typedef struct {
        size_t rs_len, hap_len;
        vector <uint8_t> q;
        vector <uint8_t> i;
        vector <uint8_t> d;
        vector <uint8_t> c;
        vector <uint8_t> hap;
        vector <uint8_t> rs;
    } testcase;
    *********************
    q >> readQuals
    i >> insrtionGOP
    d >> deletionGOP
    c >> overallGCP
    hap >> haplotypeBases  (haplen)
    rs >> readBases

*/

int main (const int argc, char const *const argv[])
{

    // Start reading data from local files
    InputReader<TestcaseIterator> testcase_reader{};
    if (argc == 2) {
        testcase_reader.from_file("argv[1]");
    } else {
        testcase_reader.from_file("tiny.in");
    }

    int counter = 0;

    for (auto &cur_testcase : testcase_reader) 
    {
        std::vector<testcase> h_input;

        int no_of_batch = cur_testcase.haplotypes.size();

        for (int i = 0; i < no_of_batch; i++) {

            counter++;

            // Transform the data format
            const auto& haplotype = cur_testcase.haplotypes[i];
            const auto& read = cur_testcase.reads[i];

            testcase new_input = {0};

            new_input.rs_len = read.bases.size();
            cout << "from_read  " << new_input.rs_len << endl;
            new_input.hap_len = haplotype.bases.size();
            cout << "from_hap  " << new_input.hap_len << endl;

            new_input.q = read.base_quals;
            new_input.i = read.ins_quals;
            new_input.d = read.del_quals;
            new_input.c = read.gcp_quals;
            new_input.hap = haplotype.bases;
            new_input.rs = read.bases;


            h_input.push_back(new_input);
        }

        double *result = g_compute_full_prob_double_gpu(h_input);
    }

    
    return 0;
}
