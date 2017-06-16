library(data.table)
nyt_details=fread("C:\\Users\\venka\\Documents\\Fall Semester\\Predictive Analytics\\Project/NYT4MSBA6420.txt")
nyt_details[1,]
names(nyt_details)=c('URL','date_collected','snippet','abstract','headline_main',
                     'pub_date','news_desk','type_of_material','id','word_count',
                     'text','facebook_like_count','facebook_share_count','googleplusone',
                     'twitter','pinterest','linkedIn')
nyt_news1=fread("C:\\Users\\venka\\Documents\\Fall Semester\\Predictive Analytics\\Project/NYT4MSBA6420text.txt",nrows = 170763)
class(nyt_news1)
names(nyt_news1)=c("URL","news")
nyt_data=merge(nyt_details,nyt_news1,by.x="URL",by.y = "URL")
nyt_data[1,]
apply(nyt_data,2,function(x) sum(is.na(x)))
sapply(nyt_data, class)
NROW(nyt_data)
write.table(nyt_data,"nyt_data.txt",sep='\t')
