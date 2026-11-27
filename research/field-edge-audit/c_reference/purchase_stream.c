/* Causal replay. Arrays are in arrival order. No reference labels enter here. */
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Precomputation depends only on preceding local scores for the same model.
   It is usable only after this head has been installed for 64 arrivals. */
void past_percentile(int n, const double *score, double *out) {
    for (int t=0;t<n;t++) {
        int begin=t>64?t-64:0, less=0, equal=0;
        for (int j=begin;j<t;j++) {
            less += score[j]<score[t];
            equal += score[j]==score[t];
        }
        out[t]=t==begin?0.5:(less+0.5*equal)/(t-begin);
    }
}

int replay(int n, int rate, int earned, int setup, int guard, int fixed,
           double cutoff, const int32_t *size, const uint8_t *source,
           const uint8_t *local, const double *score, const double *recent,
           const uint8_t *remote, uint8_t *pred, uint8_t *query, int64_t *stats) {
    const int64_t grant=(int64_t)n*rate;
    if (n<=0 || rate<=0 || setup<0 || setup>grant) return 1;
    int start=-1, calls=0;
    int64_t spent=0, review=0, max_deficit=0;
    double mean=6800.0;
    for (int t=0;t<n;t++) {
        const int64_t funded=earned?(int64_t)(t+1)*rate:grant;
        query[t]=0;pred[t]=source[t];
        if (start<0 && setup<=funded) {start=t;spent=setup;}
        if (start<0) continue;
        pred[t]=local[t]||(guard && source[t]);
        const int age=t-start;
        double priority=recent[t];
        if (start>0 && age>=16 && age<64 && !fixed) {
            int less=0,equal=0;
            for (int j=start;j<t;j++) {less+=score[j]<score[t];equal+=score[j]==score[t];}
            priority=(less+0.5*equal)/age;
        }
        double fraction=(grant-spent)/((n-t)*mean);
        if (fraction>1.0) fraction=1.0;
        int needs=fixed ? score[t]>cutoff : (age>=16 && priority>=1.0-fraction);
        if (needs && !(guard && source[t]) && size[t]<=funded-spent) {
            spent+=size[t];review+=size[t];query[t]=1;calls++;
            pred[t]=remote[t];
        }
        if (spent-funded>max_deficit) max_deficit=spent-funded;
        mean+=0.05*(size[t]-mean);
    }
    stats[0]=review;stats[1]=calls;stats[2]=start+1;stats[3]=max_deficit;
    return start<0 || spent>grant || max_deficit>0;
}
