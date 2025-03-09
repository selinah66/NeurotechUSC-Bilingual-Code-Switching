#导入数据并检查
datafile <- read.csv(file.choose(),sep = ",", dec = ".")
colnames(datafile)
str(datafile)
head(datafile)
tail(datafile)
summary(datafile)

#确定变量的正确属性
datafile$FFD = as.numeric(datafile$FFD)
datafile$GD = as.numeric(datafile$GD)
datafile$RPD = as.numeric(datafile$RPD)
datafile$TDT = as.numeric(datafile$TDT)
datafile$participant = as.factor(datafile$participant)
datafile$item = as.factor(datafile$item)
datafile$level = as.factor(datafile$level)
datafile$language = as.factor(datafile$language)
datafile$cs_types = as.factor(datafile$cs_types)
str(datafile)


#第一个自变量的系统性描述
(table1 <- ddply(datafile, .(cs_types), summarise, M=mean(FFD, na.rm = TRUE), SD=sd(FFD, na.rm = TRUE), N=length(FFD[!is.na(FFD)]), SE=SD/sqrt(N)))
#第二个自变量的系统性描述
(table1 <- ddply(datafile, .(cs_types), summarise, M=mean(GD, na.rm = TRUE), SD=sd(GD, na.rm = TRUE), N=length(GD[!is.na(GD)]), SE=SD/sqrt(N)))
#第三个自变量的系统性描述
(table1 <- ddply(datafile, .(cs_types), summarise, M=mean(RPD, na.rm = TRUE), SD=sd(RPD, na.rm = TRUE), N=length(RPD[!is.na(RPD)]), SE=SD/sqrt(N)))
#第四个自变量的系统性描述
(table1 <- ddply(datafile, .(cs_types), summarise, M=mean(TDT, na.rm = TRUE), SD=sd(TDT, na.rm = TRUE), N=length(TDT[!is.na(TDT)]), SE=SD/sqrt(N)))

# t检验
t_test_result <- t.test(FFD ~ cs_types, data = datafile)
print(t_test_result)

# t检验
t_test_result <- t.test(GD ~ cs_types, data = datafile)
print(t_test_result)

# t检验
t_test_result <- t.test(RPD ~ cs_types, data = datafile)
print(t_test_result)

# t检验
t_test_result <- t.test(TDT ~ cs_types, data = datafile)
print(t_test_result)
