# ccks2021
基于https://github.com/fuxuelinwudi/ccks2021_track3_baseline 基础上进行改动，天池F1达到85
由于模型较大没有放上来，https://ws28.cn/f/66g75r5zw0w 可以下载模型
代码里面有一些配置文件需要按照自己的设置修改
后续修改
利用FLAT进行地址分词，F1达到90%以上
预训练方式采用wwm方式
对地址进行分层匹配，融合整个地址与不同层级的匹配程度，效果提升2%
