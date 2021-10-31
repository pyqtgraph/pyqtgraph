#include <math.h>
#include <stdio.h>

void relax(
    double* pos, 
    long* links, 
    double* mrel1, 
    double* mrel2, 
    double* lengths, 
    char* push, 
    char* pull, 
    int nlinks, 
    int iters) 
    {
    int i, l, p1, p2;
    double x1, x2, y1, y2, dx, dy, dist, change;
//     printf("%d, %d\n", iters, nlinks);
    for( i=0; i<iters; i++ ) {
        for( l=0; l<nlinks; l++ ) {
            p1 = 2*links[l*2];
            p2 = 2*links[l*2 + 1];
            x1 = pos[p1];
            y1 = pos[p1 + 1];
            x2 = pos[p2];
            y2 = pos[p2 + 1];
            
            dx = x2 - x1;
            dy = y2 - y1;
            
//             dist = pow(dx*dx + dy*dy, 0.5);
            dist = sqrt(dx*dx + dy*dy);
            
            if( push[l]==0 && dist < lengths[l] )
                dist = lengths[l];
            if( pull[l]==0 && dist > lengths[l] )
                dist = lengths[l];
            
            change = (lengths[l]-dist) / dist;
            dx *= change;
            dy *= change;
        
            pos[p1]   -= mrel2[l] * dx;
            pos[p1+1] -= mrel2[l] * dy;
            pos[p2]   += mrel1[l] * dx;
            pos[p2+1] += mrel1[l] * dy;
        }
    }
}