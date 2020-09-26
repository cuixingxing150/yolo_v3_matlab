% Notice: first use "mex -setup" to choose your c/c++ compiler
clear;

%% -------------------------------------------------------------------
%% get the architecture of this computer
is_64bit = strcmp(computer,'MACI64') || strcmp(computer,'GLNXA64') || strcmp(computer,'PCWIN64');


%% -------------------------------------------------------------------
%% the configuration of compiler
% You need to modify this configuration according to your own path of OpenCV
% ע�⣺���VS OpenCVƽ̨һ��Ҫƥ��Matlab 64λ�ģ�
out_dir='./';% current folder
CPPFLAGS = ' -g -I./YoloV3Detect.h  -ID:\opencv3_4_2\opencv\build\include -ID:\opencv3_4_2\opencv\build\include\opencv -ID:\opencv3_4_2\opencv\build\include\opencv2'; % your OpenCV "include" path
LDFLAGS = ' -LD:\opencv3_4_2\opencv\build\x64\vc14\lib'; % use OpenCV release  "lib" path
LIBS = ' -lopencv_world342'; % release�汾��lib���޺�׺��ϵͳ���Զ�����ȥ
if is_64bit
    CPPFLAGS = [CPPFLAGS ' -largeArrayDims'];
end
%% add your files here!
compile_files = [
    % the list of your code files which need to be compiled
    'YoloV3Detect.cpp',' ./DetectObject.cpp'
    ];
%-------------------------------------------------------------------
%% compiling...
str = compile_files;
fprintf('compilation of: %s\n', str);
str = [str ' -outdir ' out_dir CPPFLAGS LDFLAGS LIBS];
args = regexp(str, '\s+', 'split');
mex(args{:});

fprintf('Congratulations, compilation successful!!!\n');
