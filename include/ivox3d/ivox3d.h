//
// Created by xiang on 2021/9/16.
//

#ifndef FASTER_LIO_IVOX3D_H
#define FASTER_LIO_IVOX3D_H

#include <glog/logging.h>
#include <execution>
#include <list>
#include <thread>

#include "eigen_types.h"
#include "ivox3d_node.hpp"

namespace faster_lio {

enum class IVoxNodeType {
    DEFAULT,  // linear ivox
    PHC,      // phc ivox
};

/// traits for NodeType
template <IVoxNodeType node_type, typename PointT, int dim>
struct IVoxNodeTypeTraits {};

template <typename PointT, int dim>
struct IVoxNodeTypeTraits<IVoxNodeType::DEFAULT, PointT, dim> {
    using NodeType = IVoxNode<PointT, dim>;
};

template <typename PointT, int dim>
struct IVoxNodeTypeTraits<IVoxNodeType::PHC, PointT, dim> {
    using NodeType = IVoxNodePhc<PointT, dim>;
};

//核心 IVox
template <int dim = 3, IVoxNodeType node_type = IVoxNodeType::DEFAULT, typename PointType = pcl::PointXYZ>
class IVox {
   public:
    using KeyType = Eigen::Matrix<int, dim, 1>; //voxel的xyz索引
    using PtType = Eigen::Matrix<float, dim, 1>; //点的坐标
    using NodeType = typename IVoxNodeTypeTraits<node_type, PointType, dim>::NodeType;
    using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
    using DistPoint = typename NodeType::DistPoint;

    enum class NearbyType {
        CENTER,  // center only
        NEARBY6,
        NEARBY18,
        NEARBY26,
    };

    struct Options {
        float resolution_ = 0.2;                        // ivox resolution
        float inv_resolution_ = 10.0;                   // inverse resolution
        NearbyType nearby_type_ = NearbyType::NEARBY6;  // nearby range
        std::size_t capacity_ = 1000000;                // capacity
    };

    /**
     * constructor
     * @param options  ivox options
     */
    explicit IVox(Options options) : options_(options) {
        options_.inv_resolution_ = 1.0 / options_.resolution_;
        GenerateNearbyGrids();
    }

    /**
     * add points
     * @param points_to_add
     */
    void AddPoints(const PointVector& points_to_add);

    /// get nn
    bool GetClosestPoint(const PointType& pt, PointType& closest_pt);

    /// get nn with condition
    bool GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num = 5, double max_range = 5.0);

    /// get nn in cloud
    bool GetClosestPoint(const PointVector& cloud, PointVector& closest_cloud);

    /// get number of points
    size_t NumPoints() const;

    /// get number of valid grids
    size_t NumValidGrids() const;

    /// get statistics of the points
    std::vector<float> StatGridPoints() const;

   private:
    /// generate the nearby grids according to the given options
    void GenerateNearbyGrids();

    /// position to grid，返回三维位置对应的栅格坐标
    KeyType Pos2Grid(const PtType& pt) const;

    Options options_;
    //将一个eigen矩阵映射为一个缓存内的pair数据结构<KeyType, NodeType>
    std::unordered_map<KeyType, typename std::list<std::pair<KeyType, NodeType>>::iterator, hash_vec<dim>>
        grids_map_;                                        // voxel hash map
    std::list<std::pair<KeyType, NodeType>> grids_cache_;  // voxel cache, 体素缓存
    std::vector<KeyType> nearby_grids_;                    // nearbys ， 写死一个附近的三维数据grid
};

template <int dim, IVoxNodeType node_type, typename PointType>
bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointType& pt, PointType& closest_pt) {
    std::vector<DistPoint> candidates;
    auto key = Pos2Grid(ToEigen<float, dim>(pt));
    std::for_each(nearby_grids_.begin(), nearby_grids_.end(), [&key, &candidates, &pt, this](const KeyType& delta) {
        auto dkey = key + delta;
        auto iter = grids_map_.find(dkey);
        if (iter != grids_map_.end()) {
            DistPoint dist_point;
            bool found = iter->second->second.NNPoint(pt, dist_point);
            if (found) {
                candidates.emplace_back(dist_point);
            }
        }
    });

    if (candidates.empty()) {
        return false;
    }

    auto iter = std::min_element(candidates.begin(), candidates.end());
    closest_pt = iter->Get();
    return true;
}

//主要用这个函数来搜索
template <int dim, IVoxNodeType node_type, typename PointType>
bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num,
                                                      double max_range) {
    //维护一个DistPoint的数组，大小为最多搜索的点 × 搜索的grid数量
    std::vector<DistPoint> candidates;
    candidates.reserve(max_num * nearby_grids_.size());
    // 获取搜索点的voxel索引
    auto key = Pos2Grid(ToEigen<float, dim>(pt));

// #define INNER_TIMER
#ifdef INNER_TIMER
    static std::unordered_map<std::string, std::vector<int64_t>> stats;
    if (stats.empty()) {
        stats["knn"] = std::vector<int64_t>();
        stats["nth"] = std::vector<int64_t>();
    }
#endif
    // 对所有的搜索grid进行一遍KNN
    for (const KeyType& delta : nearby_grids_) {
        auto dkey = key + delta;
        auto iter = grids_map_.find(dkey);
        if (iter != grids_map_.end()) {
#ifdef INNER_TIMER
            auto t1 = std::chrono::high_resolution_clock::now();
#endif
            auto tmp = iter->second->second.KNNPointByCondition(candidates, pt, max_num, max_range);
#ifdef INNER_TIMER
            auto t2 = std::chrono::high_resolution_clock::now();
            auto knn = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            stats["knn"].emplace_back(knn);
#endif
        }
    }

    if (candidates.empty()) {
        return false;
    }

#ifdef INNER_TIMER
    auto t1 = std::chrono::high_resolution_clock::now();
#endif
    //如果candidates的size大于max_num则要按照距离筛选
    if (candidates.size() <= max_num) {
    } else {
        std::nth_element(candidates.begin(), candidates.begin() + max_num - 1, candidates.end());
        candidates.resize(max_num);
    }
    std::nth_element(candidates.begin(), candidates.begin(), candidates.end());

#ifdef INNER_TIMER
    auto t2 = std::chrono::high_resolution_clock::now();
    auto nth = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    stats["nth"].emplace_back(nth);

    constexpr int STAT_PERIOD = 100000;
    if (!stats["nth"].empty() && stats["nth"].size() % STAT_PERIOD == 0) {
        for (auto& it : stats) {
            const std::string& key = it.first;
            std::vector<int64_t>& stat = it.second;
            int64_t sum_ = std::accumulate(stat.begin(), stat.end(), 0);
            int64_t num_ = stat.size();
            stat.clear();
            std::cout << "inner_" << key << "(ns): sum=" << sum_ << " num=" << num_ << " ave=" << 1.0 * sum_ / num_
                      << " ave*n=" << 1.0 * sum_ / STAT_PERIOD << std::endl;
        }
    }
#endif

    closest_pt.clear();
    for (auto& it : candidates) {
        closest_pt.emplace_back(it.Get());
    }
    return closest_pt.empty() == false;
}

template <int dim, IVoxNodeType node_type, typename PointType>
size_t IVox<dim, node_type, PointType>::NumValidGrids() const {
    return grids_map_.size();
}

//生成附近的grids，分别为6,18,26
template <int dim, IVoxNodeType node_type, typename PointType>
void IVox<dim, node_type, PointType>::GenerateNearbyGrids() {
    if (options_.nearby_type_ == NearbyType::CENTER) {
        nearby_grids_.emplace_back(KeyType::Zero());
    } else if (options_.nearby_type_ == NearbyType::NEARBY6) {
        nearby_grids_ = {KeyType(0, 0, 0),  KeyType(-1, 0, 0), KeyType(1, 0, 0), KeyType(0, 1, 0),
                         KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1)};
    } else if (options_.nearby_type_ == NearbyType::NEARBY18) {
        nearby_grids_ = {KeyType(0, 0, 0),  KeyType(-1, 0, 0), KeyType(1, 0, 0),   KeyType(0, 1, 0),
                         KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1),   KeyType(1, 1, 0),
                         KeyType(-1, 1, 0), KeyType(1, -1, 0), KeyType(-1, -1, 0), KeyType(1, 0, 1),
                         KeyType(-1, 0, 1), KeyType(1, 0, -1), KeyType(-1, 0, -1), KeyType(0, 1, 1),
                         KeyType(0, -1, 1), KeyType(0, 1, -1), KeyType(0, -1, -1)};
    } else if (options_.nearby_type_ == NearbyType::NEARBY26) {
        nearby_grids_ = {KeyType(0, 0, 0),   KeyType(-1, 0, 0),  KeyType(1, 0, 0),   KeyType(0, 1, 0),
                         KeyType(0, -1, 0),  KeyType(0, 0, -1),  KeyType(0, 0, 1),   KeyType(1, 1, 0),
                         KeyType(-1, 1, 0),  KeyType(1, -1, 0),  KeyType(-1, -1, 0), KeyType(1, 0, 1),
                         KeyType(-1, 0, 1),  KeyType(1, 0, -1),  KeyType(-1, 0, -1), KeyType(0, 1, 1),
                         KeyType(0, -1, 1),  KeyType(0, 1, -1),  KeyType(0, -1, -1), KeyType(1, 1, 1),
                         KeyType(-1, 1, 1),  KeyType(1, -1, 1),  KeyType(1, 1, -1),  KeyType(-1, -1, 1),
                         KeyType(-1, 1, -1), KeyType(1, -1, -1), KeyType(-1, -1, -1)};
    } else {
        LOG(ERROR) << "Unknown nearby_type!";
    }
}

template <int dim, IVoxNodeType node_type, typename PointType>
bool IVox<dim, node_type, PointType>::GetClosestPoint(const PointVector& cloud, PointVector& closest_cloud) {
    std::vector<size_t> index(cloud.size());
    for (int i = 0; i < cloud.size(); ++i) {
        index[i] = i;
    }
    closest_cloud.resize(cloud.size());

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&cloud, &closest_cloud, this](size_t idx) {
        PointType pt;
        if (GetClosestPoint(cloud[idx], pt)) {
            closest_cloud[idx] = pt;
        } else {
            closest_cloud[idx] = PointType();
        }
    });
    return true;
}

template <int dim, IVoxNodeType node_type, typename PointType>
void IVox<dim, node_type, PointType>::AddPoints(const PointVector& points_to_add) {
    //unseq用于指定并发运行，可能会加速？
    std::for_each(std::execution::unseq, points_to_add.begin(), points_to_add.end(), [this](const auto& pt) {
        auto key = Pos2Grid(ToEigen<float, dim>(pt));

        auto iter = grids_map_.find(key);
        //如果新插入的点不在原先的voxel里，则构建一个新的key和对应的node
        if (iter == grids_map_.end()) {
            PointType center;
            center.getVector3fMap() = key.template cast<float>() * options_.resolution_;
            //缓存添加新的voxel
            grids_cache_.push_front({key, NodeType(center, options_.resolution_)});
            grids_map_.insert({key, grids_cache_.begin()});

            grids_cache_.front().second.InsertPoint(pt);
            //如果缓存大小达到阈值，那么就将back的key给删掉
            //如果没有，则插入list顶端和unordered_map里
            if (grids_map_.size() >= options_.capacity_) {
                grids_map_.erase(grids_cache_.back().first);
                grids_cache_.pop_back();
            }
        } else {
        //如果新插入的点在已有的voxel内，则将该node从原先的list里移动到list前面
        //并更新key对应的位置
            iter->second->second.InsertPoint(pt);
            grids_cache_.splice(grids_cache_.begin(), grids_cache_, iter->second);
            grids_map_[key] = grids_cache_.begin();
        }
    });
}

//将三维点的xyz坐标/resolution，再round得到其在哪个三维空间voxel id上
template <int dim, IVoxNodeType node_type, typename PointType>
Eigen::Matrix<int, dim, 1> IVox<dim, node_type, PointType>::Pos2Grid(const IVox::PtType& pt) const {
    return (pt * options_.inv_resolution_).array().round().template cast<int>();
}

template <int dim, IVoxNodeType node_type, typename PointType>
std::vector<float> IVox<dim, node_type, PointType>::StatGridPoints() const {
    int num = grids_cache_.size(), valid_num = 0, max = 0, min = 100000000;
    int sum = 0, sum_square = 0;
    for (auto& it : grids_cache_) {
        int s = it.second.Size();
        valid_num += s > 0;
        max = s > max ? s : max;
        min = s < min ? s : min;
        sum += s;
        sum_square += s * s;
    }
    float ave = float(sum) / num;
    float stddev = num > 1 ? sqrt((float(sum_square) - num * ave * ave) / (num - 1)) : 0;
    return std::vector<float>{valid_num, ave, max, min, stddev};
}

}  // namespace faster_lio

#endif
