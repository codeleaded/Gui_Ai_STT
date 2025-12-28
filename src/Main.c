#include "/home/codeleaded/System/Static/Library/WindowEngine1.0.h"
#include "/home/codeleaded/System/Static/Library/GSprite.h"
#include "/home/codeleaded/System/Static/Library/NeuralNetwork.h"
#include "/home/codeleaded/System/Static/Library/Audio.h"

#define SPRITE_DATA                 "/home/codeleaded/Data/STT"
#define SPRITE_TEST                 "testing"
#define SPRITE_TRAINING             "training"
#define SPRITE_COUNT                1
#define SPRITE_MAX                  20
#define SPRITE_PATHTYPE             "wav"

#define NN_PATH                     "./data/Model.nnalx"
#define NN_COUNT                    10
#define NN_LEARNRATE                0.5f

#define AUDIO_SAMPLE_RATE           44100
#define AUDIO_CHANNELS              1
#define AUDIO_BITS_PER_SAMPLE       16
#define AUDIO_FORMAT                SND_PCM_FORMAT_S16_LE
#define AUDIO_DURATION_SECONDS      5
#define AUDIO_FRAMES_PER_BUFFER     1024
#define AUDIO_WORD_ROT              0
#define AUDIO_WORD_GRUEN            1
#define AUDIO_WORD_BLAU             2
#define AUDIO_WORD_WHEISS           3
#define AUDIO_WORD_SCHWARZ          4
#define AUDIO_WORD_GRAU             5
#define AUDIO_WORD_GELB             6
#define AUDIO_WORD_LILA             7
#define AUDIO_WORD_TURKIS           8
#define AUDIO_WORD_ORANGE           9


int epoch = 0;
int reality = 0;
int prediction = 0;
NeuralType loss = 0.0f;
NeuralNetwork nnet;

int created[NN_COUNT];
int datapoint = 0;
IAudio micro;


NeuralDataPair NeuralDataPair_Make_Audio(char* path,int number,int item,int output){
    CStr ntraining_s = CStr_Format("%s/%d/%d." SPRITE_PATHTYPE,path,number,item);
    WavFile wf = WavFile_Read(ntraining_s,micro.bits / 8 * micro.channels);
    CStr_Free(&ntraining_s);
    
    if(output > 0 && wf.buffer){
        NeuralType* outs = (NeuralType*)malloc(sizeof(NeuralType) * output);
        memset(outs,0,sizeof(NeuralType) * output);
        outs[number] = 1.0f;

        const int size = IAudio_Size(&micro,1000000);
        NeuralType* data = (NeuralType*)malloc(sizeof(NeuralType) * size);
        for(int i = 0;i<size;i++){
            const short sample = *(short*)wf.buffer + i;
            const NeuralType nts = (NeuralType)sample / (NeuralType)0xFFFF + 0.5f;
            data[i] = nts;
        }

        NeuralDataPair ndp = NeuralDataPair_Move(data,outs,size,output);
        WavFile_Free(&wf);
        return ndp;
    }else{
        printf("[NeuralDataPair]: Load -> '%s/%d/%d." SPRITE_PATHTYPE "' not avalible!\n",path,number,item);
    }
    
    WavFile_Free(&wf);
    return NeuralDataPair_Null();
}
NeuralDataMap NeuralDataMap_Make_Audio(char* path,int* epoch,int output,int count,int maxcount){
    NeuralDataMap ndm = NeuralDataMap_New();
    for(int i = 0;i<output;i++){
        for(int j = 0;j<count;j++){
            NeuralDataPair ndp = NeuralDataPair_Make_Audio(path,i,*epoch + j,output);
            Vector_Push(&ndm,&ndp);
        }
    }

    if(epoch){
        *epoch += count;
        if(*epoch + count > maxcount)
            *epoch = 0;
    }
    return ndm;
}

void Setup(AlxWindow* w){
    nnet = NeuralNetwork_Make((NeuralLayerBuilder[]){
        NeuralLayerBuilder_Make(44100,"relu"),
        NeuralLayerBuilder_Make(16,"relu"),
        NeuralLayerBuilder_Make(NN_COUNT,"softmax"),
        NeuralLayerBuilder_End()
    });

    //IAudio a = IAudio_New(AUDIO_FORMAT,AUDIO_BITS_PER_SAMPLE,AUDIO_FRAMES_PER_BUFFER,2,AUDIO_SAMPLE_RATE);
    //WavFile wf = WavFile_Read(NULL,AUDIO_FRAMES_PER_BUFFER);
    //WavFile_Print(&wf);
    //OAudio_Adapt(&a,&wf);
    //OAudio_Play(&a,&wf);
    //WavFile_Free(&wf);
    //OAudio_Free(&a);

    memset(created,0,sizeof(created));
    micro = IAudio_New(AUDIO_FORMAT,AUDIO_BITS_PER_SAMPLE,AUDIO_FRAMES_PER_BUFFER,AUDIO_CHANNELS,AUDIO_SAMPLE_RATE,500000);
    //IAudio_Write(&a,NULL);
}
void Update(AlxWindow* w){
    if(micro.running){
        if(Time_ElapsedD(micro.start,Time_Nano()) > TIME_NANOTOSEC){
            IAudio_Stop(&micro);
            IAudio_ClipDuration(&micro,1000000);

            CStr name = CStr_Format(SPRITE_DATA "/" SPRITE_TRAINING "/%d/%d." SPRITE_PATHTYPE,datapoint,created[datapoint]++);
            IAudio_Write(&micro,name);
            CStr_Free(&name);
        }
    }
    
    if(Stroke(ALX_KEY_Q).PRESSED){
        NeuralNetwork_Save(&nnet,NN_PATH);
        printf("[NeuralNetwork]: Save -> Success!\n");
    }else if(Stroke(ALX_KEY_E).PRESSED){
        if(Files_isFile(NN_PATH)){
            NeuralNetwork_Free(&nnet);
            nnet = NeuralNetwork_Load(NN_PATH);
            printf("[NeuralNetwork]: Load -> Success!\n");
        }else{
            printf("[NeuralNetwork]: Load -> Failed!\n");
        }
    }else if(Stroke(ALX_KEY_1).PRESSED){
        datapoint--;
        if(datapoint < 0)
            datapoint = NN_COUNT - 1;
    }else if(Stroke(ALX_KEY_2).PRESSED){
        datapoint++;
        if(datapoint >= NN_COUNT)
            datapoint = 0;
    }else if(Stroke(ALX_KEY_D).PRESSED){
        if(!micro.running){
            IAudio_Clear(&micro);
            micro.start = 0UL;
            micro.duration = 0UL;
            IAudio_Start(&micro);
        }
    }else if(Stroke(ALX_KEY_W).PRESSED){
        NeuralDataMap ndm = NeuralDataMap_Make_Audio(SPRITE_DATA "/" SPRITE_TRAINING,&epoch,NN_COUNT,SPRITE_COUNT,SPRITE_MAX);
        NeuralNetwork_Learn(&nnet,&ndm,NN_LEARNRATE);
        loss = NeuralNetwork_Test_C(&nnet,&ndm);
        NeuralDataMap_Free(&ndm);
    }else if(Stroke(ALX_KEY_S).PRESSED){
        unsigned int ndir = 0;//Random_u32_MinMax(0,NN_COUNT);
        unsigned int item = Random_u32_MinMax(0,SPRITE_MAX);

        NeuralDataPair ndp = NeuralDataPair_Make_Audio(SPRITE_DATA "/" SPRITE_TRAINING,ndir,item,NN_COUNT);
        loss = NeuralNetwork_Test(&nnet,&ndp);
        NeuralDataPair_Free(&ndp);

        prediction = NeuralNetwork_Decision(&nnet);
        reality = ndir;

        CStr ntraining_s = CStr_Format(SPRITE_DATA "/" SPRITE_TEST "/%d/%d." SPRITE_PATHTYPE,ndir,item);
        CStr_Free(&ntraining_s);
    }

    Clear(DARK_BLUE);

    const int padding = 50;
    const int bwidth = 5;
    const int count = (GetWidth() - 2 * padding) / bwidth;
    for(int i = 0;i<count;i++){
        const short sampleindex = i * IAudio_Size(&micro,1000000) / (count * sizeof(short));
        if(sampleindex >= micro.buffer.size / sizeof(short) || !micro.buffer.Memory) break;
        
        const short sample = *((short*)micro.buffer.Memory + sampleindex);
        const int h = I32_Abs(sample) * (int)((float)GetHeight() * 0.45f) * 2 / 0x7FFF;
        const int x = padding + i * bwidth;
        const int y = GetHeight() / 2 - h / 2;
        Rect_RenderXX(WINDOW_STD_ARGS,x,y,bwidth,h,RED);
    }

    CStr_RenderAlxFontf(WINDOW_STD_ARGS,GetAlxFont(),0.0f,0.0f,WHITE,"DP: %d, Len: %d",(Number)datapoint,(Number)micro.buffer.size);
    CStr_RenderAlxFontf(WINDOW_STD_ARGS,GetAlxFont(),0.0f,GetHeight() - GetAlxFont()->CharSizeY,WHITE,"Loss: %f, Is: %d, Pre: %d, -> %s",loss,reality,prediction,(reality == prediction ? "correct" : "wrong"));
}
void Delete(AlxWindow* w){
    IAudio_Free(&micro);
    NeuralNetwork_Free(&nnet);
}

int main(){
    if(Create("RGB to G",1920,1080,1,1,Setup,Update,Delete))
        Start();
    return 0;
}