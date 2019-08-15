#include "infer_yellow.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace yellow;
//Get all image name
bool getImageName(const char *fileName, vector<string> &imageName)
{
    FILE *f = fopen(fileName, "r");
    if (f == NULL)
        return false;
    char buffer[300];
    while (fgets(buffer, 300, f))
    {
        //Cut '\n'
        int len = strlen(buffer);
        if(buffer[len - 1] == '\n')
            buffer[strlen(buffer) - 1] = '\0';
        imageName.push_back(string(buffer));
    }
    fclose(f);
    return true;
}

int main(int argc, char const *argv[]) 
{

    if(argc<4)
    {
        std::cout<<"error : miss some models."<<endl;
        return 0;
    }
    //output the models of inputs
    for(int i=1;i<argc;i++)
    {
        std::cout<<argv[i]<<std::endl;
    }
    //the batch of model
    const int batch = 8;
    std::vector<std::string> models;
    //u can load multi models
    {
        models.push_back(argv[1]);
        models.push_back(argv[2]);
        models.push_back(argv[3]);
    }

    int code = -1;
    char *err = nullptr;
    int gpuid = 0;
    // Do createNet
    auto *handle = createNet(models, gpuid, &code, &err);

    if (code == 200) 
    {   
        vector<string> imageName;
        // Judge images can be open
        if (!getImageName("../images/file.txt", imageName))
        {
            cerr << "Can't open image" << endl;
        }
        int num_times = imageName.size()/batch;

        //deal with the images not matches
        bool full_batch=true;
        if(num_times*batch!=imageName.size())
        {
            num_times++;
            full_batch=false;
        }
        //Get the image number of each inference 
        int each_time_images=batch;
        //Do inference
        for (int i = 0; i < num_times; ++i) 
        {
            // Read image from file to bytes
            vector<cv::Mat> imgs;
            //Get image index of each inference.
            
            if(!full_batch && i==num_times-1)
                each_time_images=imageName.size()%batch;
            for (int n = 0; n < each_time_images; n++) 
            {
                string name = imageName[i * batch + n];
                cv::Mat img = cv::imread("../images/" + name);
                if (!img.data)
                {
                    cerr << "Read image " << name << " error, No Data!" << endl;
                    continue;
                }
                imgs.push_back(img);
            }
  
            vector<string> results;
            
            //Infer body
            netInference(handle, imgs, &code, &err,&results);

            if (code == 200) 
            {
                //Get the results of each infer. 
                for (int i = 0; i < imgs.size(); i++)
                    {
                        std::cout <<results.at(i) << std::endl;
                    }
            } 
            else 
            {
                std::cout << err << std::endl;
            }
        }
    } 
    else 
    {
        std::cout << err << std::endl;
    }
    //release variables space
    releaseNet(handle,&code,&err);
    return 0;
}
