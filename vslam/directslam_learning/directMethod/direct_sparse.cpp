#include <iostream>
#include <fstream>//读写文件
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>


#include <g2o/core/base_unary_edge.h>  //边定义的头文件
#include <g2o/core/block_solver.h> //解海塞矩阵块
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h> 
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;
using namespace g2o;

//定义一个测量值结构体，里面包括一个三维世界的坐标，一个灰度
struct Measurement
{
    Measurement ( Eigen::Vector3d p ,float g) : pos_world ( p ),grayscale ( g ) {}
    Eigen::Vector3d pos_world;
    float grayscale;    
};

//根据小孔成像模型反推，2D点到3D点的变换
inline Eigen::Vector3d project2Dto3D( int x, int y, int d, float fx, float fy, float cx, float cy, float scale)
{
    float zz = float (d) / scale;
    float xx = zz* ( x - cx ) / fx;
    float yy = zz* ( y - cy ) / fx;
    
    return Eigen::Vector3d (xx,yy,zz);
}

//根据小孔成像模型，3D点到2D点的变换
inline Eigen::Vector2d project3Dto2D( float x, float y,float z, float fx, float fy, float cx, float cy)
{
    float u = fx * x / z + cx;
    float v = fy * y / z + cy;
    return Eigen::Vector2d(u,v);
}

// 直接法估计位姿
//声明函数
// 直接法估计位姿
// 输入：测量值（空间点的灰度），新的灰度图，相机内参； 输出：相机位姿
// 返回：true为成功，false失败
bool poseEstimationDirect ( const vector<Measurement>& measurements, cv::Mat* gray, Eigen::Matrix3f& intrinsics, Eigen::Isometry3d& Tcw);



//定义一个类,由基类EaseUnaryEdge(g2o中的边)继承而来,
//VertexSE3Expmap是相机位姿变化的李代数描述，作为图优化的顶点(Vertex)
class EdgeSE3ProjectDirect: public BaseUnaryEdge <1,double,VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EdgeSE3ProjectDirect(){} //构造函数1
    
    EdgeSE3ProjectDirect( Eigen::Vector3d point, float fx,float fy, float cx, float cy, cv::Mat* image) : x_world_( point ),fx_( fx ), fy_( fy ),cx_( cx ), cy_( cy ), image_ ( image ) {}   //构造函数2，参数是一个3D点坐标，相机内参，cv的图象
    
    virtual  void computeError() //计算误差，虚函数
    {
        const VertexSE3Expmap* v = static_cast<const VertexSE3Expmap*> (_vertices[0]) ;//定义一个相机位姿的变换
        Eigen::Vector3d x_local = v->estimate().map (x_world_);
        
        float x = x_local[0] * fx_ / x_local[2] + cx_;
        float y = x_local[1] * fy_ / x_local[2] + cy_;
        //将3D点转换到图象上
        
        //先检查该点是否在图像上，如果不在，误差就为0，如果在，就计算误差
        if( x-4 <0 || (x+4) > image_->cols || y-4 <0 || (y+4) > image_->rows)
        {
            _error (0,0) = 0.0;
            this -> setLevel (1);//在optimizable_graph.h的435行，应该是设置等级标志的
        }
        else
        {
            _error (0,0) = getPixeValue(x,y) - _measurement;
        }
        
    }
    
    virtual void linearizeOplus()//重写计算雅可比矩阵
    {
        if ( level() == 1 )//如果点不在图像内,雅可比矩阵就全为0
        {
            _jacobianOplusXi = Eigen::Matrix<double,1,6>::Zero();
            return;
        }
        VertexSE3Expmap* vtx = static_cast<VertexSE3Expmap*> (_vertices[0]) ;//定义一个相机位姿的变换
        Eigen::Vector3d xyz_trans = vtx->estimate().map (x_world_);
        ////书P194的q,扰动分量在第二个相机坐标系下的坐标
/*
    * Eigen::Vector3d map (const Eigen::Vector3d& xyz) const {return s*(r*xyz) + t;}
    */
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0/xyz_trans[2];//z是倒数
        double invz_2 = invz * invz ;  //z的平方
        
        float u = x * fx_ * invz + cx_ ;
        float v = y * fy_ * invz + cy_ ;
        
        
        
        Eigen::Matrix<double,2,6> jacobian_uv_ksai;
        ////书P195的公式8.15的2*6矩阵，uv像素坐标系对ksai求偏导，
        //要注意的是在g2o中的李代数的矩阵排序与书中不同，，左右三列分别反过来
        
        jacobian_uv_ksai ( 0,0 ) = - x*y*invz_2 *fx_;
        jacobian_uv_ksai ( 0,1 ) = ( 1+ ( x*x*invz_2 ) ) *fx_;
        jacobian_uv_ksai ( 0,2 ) = - y*invz *fx_;
        jacobian_uv_ksai ( 0,3 ) = invz *fx_;
        jacobian_uv_ksai ( 0,4 ) = 0;       
        jacobian_uv_ksai ( 0,5 ) = -x*invz_2 *fx_;
                                            
        jacobian_uv_ksai ( 1,0 ) = - ( 1+y*y*invz_2 ) *fy_;
        jacobian_uv_ksai ( 1,1 ) = x*y*invz_2 *fy_;
        jacobian_uv_ksai ( 1,2 ) = x*invz *fy_;
        jacobian_uv_ksai ( 1,3 ) = 0;       
        jacobian_uv_ksai ( 1,4 ) = invz *fy_;
        jacobian_uv_ksai ( 1,5 ) = -y*invz_2 *fy_;
        
        Eigen::Matrix<double,1,2> jacobian_pixel_uv;
        //图像在该点处的梯度，差分法
        jacobian_pixel_uv (0,0) = ( getPixeValue (u+1,v) - getPixeValue (u-1,v) )/2;
        jacobian_pixel_uv (0,1) = ( getPixeValue (u,v+1) - getPixeValue (u,v-1) )/2;
        
        _jacobianOplusXi = jacobian_pixel_uv*jacobian_uv_ksai;
        //两者相乘得到雅可比矩阵，P195的式8.16，这里没有负号是应为误差计算是用的现在图片的亮度减前一图片的亮度，跟书中是相反数
    }
        
        // dummy read and write functions because we don't care...
        virtual bool read ( std::istream& in ) {}        
        virtual bool write ( std::ostream& out ) const {}        
    protected:
        //使用双线性插值法计算参考帧的某一点灰度直
        inline float getPixeValue(float x,float y)
        {
            uchar* data = & image_ ->data[int ( y )* image_ -> step + int ( x )];
            float xx = x - floor ( x );
            float yy = y - floor ( y );
            
            return float (
                            ( 1 -xx ) * ( 1- yy ) * data[0] +
                            xx * ( 1 - yy ) * data[1] +
                            ( 1 - xx ) * yy * data[ image_->step ] +
                            xx * yy *data[ image_->step+1 ] 
                
            );
            
        }
        
    public:
        Eigen::Vector3d x_world_;  // 3D point in world frame
        float cx_ = 0 , cy_ = 0 , fx_ = 0 , fy_ = 0 ; // Camera intrinsics
        cv::Mat* image_ = nullptr;  // reference image
  
    
};



int main(int argc,char ** argv)
{
    if( argc != 2)
    {
        cout << "usage: useLK path_to_dataset" << endl;
        return 0;
    }
    
    srand((unsigned int)time(0));
    string path_to_dataset = argv[1];//数据集的路径
    string associate_file = path_to_dataset + "/associate.txt";
    
    ifstream fin (associate_file);//定义一个associate_file的输入流
    
    string rgb_file, depth_file, time_rgb, time_depth;
    
    cv::Mat color,depth,gray;
    vector<Measurement> measurements;//定义一个Measurement类型的向量measurements
    // 相机内参      
    float cx = 325.5;
    float cy = 253.5;                                                  
    float fx = 518.0;
    float fy = 519.0;
    float depth_scale = 1000.0;//?????? 深度的尺度

    Eigen::Matrix3f K;
    K<<fx,0.f,cx,0.f,fy,cy,0.f,0.f,1.0f;//内参矩阵

    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
    /*定义了Tcw为Isometry3d类型的旋转
    Isometry
      A transformation that is invariant with respect to distance. That is, the distance between any two points in the pre-image must be the same as the distance between the images of the two points.
    */
    
    cv::Mat prev_color;
    
    
    // 对十张图片作直接法估计位姿，我们以第一个图像为参考，对后续图像和参考图像做稀疏直接法
    for( int index = 0; index < 10; index ++)
    {
        cout<<"*********** loop "<<index<<" ************"<<endl;
        fin >> time_rgb >> rgb_file >>  time_depth >> depth_file;
        
        color = cv::imread(path_to_dataset + "/"+rgb_file);
        depth = cv::imread(path_to_dataset + "/"+depth_file,-1);//flags<0时表示以图片的本来的格式读入图片
        
        if( color.data ==nullptr || depth.data== nullptr)
            continue;
        cv::cvtColor(color,gray,cv::COLOR_BGR2GRAY);//将BGR图片转化为灰度图
        if( index == 0)
        {
            
            //对第一帧图片提取FAST特征点
            vector<cv::KeyPoint> Keypoints;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();//创建快速角点
            //寻找FAST角点，
            /*FAST-N 角点的定义
             * 
             * 1.定义一个像素p的亮度为Ip
             * 2.定义一个阈值为T
             * 3.选取以p为圆心,3为半径的16个像素点，如果连续的N个像素的亮度都大于Ip+T或小于Ip-T，那么就定义这个点为角点，通常N=12，称为FAST-12
            */
            
            detector->detect(color,Keypoints);//寻找角点
            
            for( auto kp:Keypoints)
            {
                //滤掉邻近图片边缘处(20以内)的角点
                if( kp.pt.x < 20 || kp.pt.y < 20 || ( kp.pt.x+20 ) > color.cols || ( kp.pt.y + 20 ) > color.rows )
                    continue;
               
                ushort d  = depth.ptr< ushort > ( cvRound( kp.pt.y ) ) [ cvRound ( kp.pt.x )];
                
                if ( d == 0 )
                {
                    continue;
                }
                
                Eigen::Vector3d p3d = project2Dto3D ( kp.pt.x, kp.pt.y, d, fx, fy, cx, cy, depth_scale);
                float grayscale = float ( gray.ptr<uchar> (cvRound( kp.pt.y))  [ cvRound( kp.pt.x ) ] );//获取这个点的灰度值
                measurements.push_back ( Measurement ( p3d , grayscale) );//向measurements中加入这个点
                
            }
            
            prev_color = color.clone();
            continue;         
            
        }
        
        
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        
        //使用直接法计算相机运动
        poseEstimationDirect( measurements, &gray , K ,Tcw );
        
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        
        //打印估算的时间
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
        cout<<"direct method costs time: "<<time_used.count() <<" seconds."<<endl;
        cout<<"Tcw="<<Tcw.matrix() <<endl;
        
        //在图中进行plot标注
        cv::Mat img_show ( color.rows*2 , color.cols , CV_8UC3 );
        prev_color.copyTo (img_show ( cv::Rect ( 0,0,color.cols,color.rows)));
        color.copyTo( img_show ( cv::Rect ( 0, color.rows, color.cols, color.rows )));
        for( Measurement m:measurements )//对提取的特征点作标注
        {
            if( rand()> RAND_MAX/5 )  //生成随机数
                continue;
            Eigen::Vector3d p = m.pos_world;
            Eigen::Vector2d pixel_prev = project3Dto2D ( p (0,0), p (1,0), p (2,0),   fx, fy, cx, cy );
            Eigen::Vector3d p2 = Tcw * m.pos_world;
            Eigen::Vector2d pixel_now =  project3Dto2D ( p2(0,0), p2 (1,0), p2 (2,0), fx, fy, cx, cy);
            
            if( pixel_now(0,0) < 0 || pixel_now(0,0) >= color.cols || pixel_now(1,0)<0 || pixel_now(1,0) >= color.rows )
                continue;
            
            float b = 255* float (rand()) / RAND_MAX;
            float g = 255* float (rand()) / RAND_MAX;
            float r = 255* float (rand()) / RAND_MAX;
            
            cv::circle ( img_show, cv::Point2d ( pixel_prev (0,0),  pixel_prev (1,0) ),             8 , cv::Scalar (b,g,r), 2 );
            cv::circle ( img_show, cv::Point2d ( pixel_now  (0,0),  pixel_now (1,0) + color.rows),  8 , cv::Scalar (b,g,r), 2);
            cv::line   ( img_show, cv::Point2d ( pixel_prev (0,0),  pixel_prev (1,0) ), cv::Point2d ( pixel_now  (0,0),  pixel_now (1,0) + color.rows), cv::Scalar (b,g,r), 1);
            
        }

        cv::imshow( "result", img_show );
        cv::waitKey( 0 );
        
    }    
    
    return 1;
}

bool poseEstimationDirect ( const vector<Measurement>& measurements, cv::Mat* gray, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw)
{
    //初始化g2o
/*   
 *在g2o中选择优化方法一共需要三个步骤： 
1.选择一个线性方程求解器，从 PCG, CSparse, Choldmod中选，实际则来自 g2o/solvers 文件夹
2.选择一个 BlockSolver 。
3.选择一个迭代策略，从GN, LM, Doglog中选。
*/
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;
    //求解的向量是 6*1 的，旋转R 平移T DirectBlock就是这个
    
    DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType> ();
    // 线性方程求解器：稠密的增量方程  HΔx=−b
    
    DirectBlock* solver_ptr = new DirectBlock( linearSolver ) ;
    //矩阵块求解器
    
    g2o::OptimizationAlgorithmLevenberg* slover = new g2o::OptimizationAlgorithmLevenberg(solver_ptr) ; 
    //使用LM迭代策略
    
    g2o::SparseOptimizer optimizer;   //图模型
    optimizer.setAlgorithm ( slover );//设置求解器
    optimizer.setVerbose (true);      //打开调试输出
    
    //添加顶点，位姿是顶点，是需要被优化的变量
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setEstimate ( g2o::SE3Quat ( Tcw.rotation(),Tcw.translation() ) );
    //被估计的是SE3的旋转矩阵，平移矩阵的四元数表示方法？？？
    pose->setId(0);
    optimizer.addVertex( pose );
    
    //添加边,带噪声的数据点，构成了一个个误差项，也就是边
    int id = 1;
    for (Measurement m : measurements)//把measurements中所有的已有数据都建立起很多条边到一个点的图
    {
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect(
            m.pos_world,
            K (0,0) , K (1,1) ,  K (0,2) ,  K (1,2) , gray
        );//根据已经定义的边的类，实例化一条边，并有初始化列表，
        edge->setVertex(0,pose);//设置连接的顶点
        edge->setMeasurement(m.grayscale);//观测数值
        edge->setInformation( Eigen::Matrix<double,1,1>::Identity() );//信息矩阵，协方差矩阵之逆
        edge->setId (id++);
        optimizer.addEdge(edge);
        
    }
    
    cout<<"edges in graph: "<<optimizer.edges().size() <<endl;
    optimizer.initializeOptimization();
    optimizer.optimize ( 30 ); //开始迭代
    Tcw = pose->estimate();//将pose的优化值写入Tcw
        
}
