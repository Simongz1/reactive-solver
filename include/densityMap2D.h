#pragma once
#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace densityMap2D {
enum class Phase { RDX, PS, Pore };

struct Map {
    std::vector<double> x,y,rho;
    std::vector<Phase> phases; // Populated only for explicitly phase-aware reads.
    static std::size_t nearest(const std::vector<double>& axis,double q) {
        auto it = std::lower_bound(axis.begin(),axis.end(),q);
        if(it == axis.begin()){
            return 0;
        }
        if(it == axis.end()){
            return axis.size() - 1;
        }

        const auto hi = std::size_t(it - axis.begin());

        return q - axis[hi - 1] <= axis[hi] - q ? hi - 1 : hi;
    }

    std::size_t indexAt(double px,double py) const {
        auto check = [](const std::vector<double>& a, double q) {
            const double lo = a.front() - 0.5 * (a[1] - a[0]);
            const double hi = a.back() + 0.5 * (a.back() - a[a.size() - 2]);
            if(q<lo || q>hi)throw std::invalid_argument("Density map does not cover FV cell centers");
        };
        check(x,px);check(y,py);
        return nearest(y,py)*x.size()+nearest(x,px);
    }
    double at(double px,double py) const {
        return rho[indexAt(px,py)];
    }
    Phase phaseAt(double px,double py) const {
        if(phases.empty())throw std::invalid_argument("Density map has no phase column");
        return phases[indexAt(px,py)];
    }
};
inline Map read(const std::string& path, bool requirePhase = false) {
    std::ifstream file(path);
    if(!file)throw std::invalid_argument("Cannot open density map: "+path);
    std::string line;
    if(!std::getline(file,line))throw std::invalid_argument("Empty density map");
    line.erase(std::remove_if(line.begin(),line.end(),[](unsigned char c){return std::isspace(c);}),line.end());
    const std::string expectedHeader=requirePhase?"x,y,rho,phase":"x,y,rho";
    if(line!=expectedHeader)throw std::invalid_argument("Density map header must be "+expectedHeader);
    struct Point {double x,y,rho; Phase phase=Phase::RDX;};std::vector<Point> points;Map map;
    int row=1;
    while(std::getline(file,line)) {
        ++row;if(line.find_first_not_of(" \t\r")==std::string::npos)continue;
        std::replace(line.begin(),line.end(),',',' ');std::istringstream in(line);
        Point p;std::string extra;
        if(!(in>>p.x>>p.y>>p.rho) || !std::isfinite(p.x)
            || !std::isfinite(p.y) || !std::isfinite(p.rho) || p.rho<0)
            throw std::invalid_argument("Invalid density map row "+std::to_string(row));
        if(requirePhase) {
            std::string phase;
            if(!(in>>phase))throw std::invalid_argument("Missing phase at row "+std::to_string(row));
            if(phase=="rdx")p.phase=Phase::RDX;
            else if(phase=="ps")p.phase=Phase::PS;
            else if(phase=="pore")p.phase=Phase::Pore;
            else throw std::invalid_argument("Phase must be rdx, ps or pore at row "+std::to_string(row));
        }
        if(in>>extra)throw std::invalid_argument("Extra column at density map row "+std::to_string(row));
        points.push_back(p);map.x.push_back(p.x);map.y.push_back(p.y);
    }
    for(auto* a:{&map.x,&map.y}) {
        std::sort(a->begin(),a->end());a->erase(std::unique(a->begin(),a->end()),a->end());
        if(a->size()<2)throw std::invalid_argument("Density map needs at least two x and y coordinates");
    }
    if(points.size()!=map.x.size()*map.y.size())throw std::invalid_argument("Density map must be a complete rectangular grid");
    map.rho.assign(points.size(),-1);
    if(requirePhase)map.phases.resize(points.size());
    for(const auto& p:points) {
        auto& value=map.rho[Map::nearest(map.y,p.y)*map.x.size()+Map::nearest(map.x,p.x)];
        if(value>=0)throw std::invalid_argument("Duplicate density map coordinate");
        value=p.rho;
        if(requirePhase)map.phases[Map::nearest(map.y,p.y)*map.x.size()+Map::nearest(map.x,p.x)]=p.phase;
    }
    return map;
}
}
