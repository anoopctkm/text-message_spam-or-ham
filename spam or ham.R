library(tm)
library(dplyr)
library(stringr)
library(caret)
library(caTools)

set.seed(19)
texts <- read.csv("spam.csv")

glimpse(texts)
texts <- select(texts, v1, v2)

texts$v1 <- as.character(texts$v1)
texts$v1 <- str_replace(texts$v1, "ham", "0")
texts$v1 <- str_replace(texts$v1, "spam", "1")
texts$v1 <- as.factor(texts$v1)

table(texts$v1)

texts_corp <- Corpus(VectorSource(texts$v2))

texts_corp <- texts_corp %>%
  tm_map(PlainTextDocument) %>% 
  tm_map(removePunctuation) %>% 
  tm_map(removeNumbers) %>%  
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeWords, stopwords(kind = "en")) %>% 
  tm_map(content_transformer(stripWhitespace))
  
  dtm <- DocumentTermMatrix(x = texts_corp,
                          control = 
                          list(tokenize = "words",
                               stemming = "english",
                               weighting = weightTf))
                               
 tf <- as.data.frame.matrix(dtm)
 dim(tf)
 term_freq <- colSums(tf)
freq <- data.frame(term = names(term_freq), count = term_freq)
arrange(freq, desc(count))[1:20,]

most_freq <- filter(freq, count >= 50)

tf <- select(tf, already,    
             also,    
             always,    
             amp,    
             anything,    
             around,    
             ask,    
             babe,    
             back,   
             box,    
             buy,    
             call,   
             can,   
             cant,   
             care,    
             cash,    
             chat,    
             claim,   
             come,   
             coming,    
             contact,    
             cos,    
             customer,    
             day,   
             dear,   
             didnt,    
             dont,   
             dun,    
             even,    
             every,    
             feel,    
             find,    
             first,    
             free,   
             friends,    
             get,   
             getting,    
             give,   
             going,   
             gonna,    
             good,   
             got,   
             great,   
             guaranteed,    
             gud,    
             happy,   
             help,    
             hey,   
             home,   
             hope,   
             ill,   
             ive,    
             just,   
             keep,    
             know,   
             last,    
             late,    
             later,   
             leave,    
             let,    
             life,    
             like,   
             lol,    
             lor,   
             love,   
             ltgt,   
             make,   
             many,    
             meet,    
             message,    
             mins,    
             miss,    
             mobile,   
             money,    
             morning,    
             msg,    
             much,   
             name,    
             need,   
             new,   
             nice,    
             night,   
             nokia,    
             now,   
             number,    
             one,   
             people,    
             per,    
             phone,   
             pick,    
             place,    
             please,   
             pls,   
             prize,    
             really,    
             reply,   
             right,    
             said,    
             say,    
             see,   
             send,   
             sent,    
             service,    
             sleep,    
             someone,    
             something,    
             soon,    
             sorry,   
             still,   
             stop,   
             sure,    
             take,   
             tell,   
             text,   
             thanks,    
             thats,    
             thing,    
             things,    
             think,   
             thk,    
             time,   
             today,   
             told,    
             tomorrow,    
             tonight,    
             txt,   
             urgent,    
             wait,    
             waiting,    
             wan,    
             want,   
             wat,    
             way,   
             week,   
             well,   
             went,    
             will,   
             win,    
             wish,    
             won,    
             wont,    
             work,    
             yeah,    
             year,    
             yes,    
             yet,    
             youre)
             
texts_tf <- mutate(tf, Class = texts$v1)
dim(texts_tf)
trainIndex <- createDataPartition(texts_tf$Class, 
                                  times = 1,
                                  p = 0.8, 
                                  list = FALSE)

train <- texts_tf[trainIndex, ]
test <- texts_tf[-trainIndex, ]

model <- train(Class ~ .,
               data = train,
               method = "glm",
               trControl = trainControl(method = "cv",
                                        number = 10)
               )

model

varImp(model)
p <- predict(model, test, type = "prob")

predictions <- factor(ifelse(p["0"] > 0.4,
                              "0",
                              "1"))

confusionMatrix(predictions, test$Class)

colAUC(p, test$Class, plotROC = TRUE)
