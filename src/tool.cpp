#include "tools.hpp"
#include<cmath>
#include<iostream>
#include<algorithm>

double mean(const std::vector<double>& values){
    double sum=0.0;

    for(double v:values)
    {
        sum+=v;
    }
    return sum/values.size();
};

double findp(std::vector<double> data,double p)
{
    if(data.empty())
        return 0.0;
    size_t n=data.size();
    double rank=p*(n-1);
    size_t lo = static_cast<size_t>(std::floor(rank));
    size_t hi = static_cast<size_t>(std::ceil(rank));

    std::nth_element(data.begin(),data.begin()+lo,data.end());
    double lon=data[lo];

    if(lo==hi)
        return lon;

    std::nth_element(data.begin(),data.begin()+hi,data.end());
    double hin=data[hi];

    return hin-(hin-lon)*(hi-rank);
   
};

double findmin(std::vector<double> data)
{
    if(data.empty())
        return 0.0;

    std::nth_element(data.begin(),data.begin(),data.end());
    double min=data[0];

    return min;
};

double findmax(std::vector<double> data)
{
    if(data.empty())
        return 0.0;

    std::nth_element(data.begin(),data.end()-1,data.end());
    double max=data[999];

    return max;
};

void coutime(std::vector<double> data)
{   
    std::cout
        << "mean: " <<mean(data) << " ms\n";
    std::cout
        <<"p50:  "<<findp(data,0.5) << " ms\n";
    std::cout
        <<"p95:  "<<findp(data,0.95) << " ms\n";
    std::cout
        <<"min:  "<<findmin(data) << " ms\n";
    std::cout
        <<"max:  "<<findmax(data) << " ms\n";
};