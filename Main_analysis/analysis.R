
# Replication script for Bollen, Higton and Sands: Nationally Representative, Locally Misaligned

# Code comments are hierarchical: 
# RStudio users can click on the contents symbol (next to source, top right) or at bottom (above console/terminal)
# This shows the complete contents/structure of the script

library(knitr)
library(sf)
library(kableExtra)
library(haven)
library(broom)
library(tidyverse)
library(osmdata)
library(psych)
library(patchwork)
library(srvyr)
library(xtable)
library(weights)
library(texreg)

# Set wd as current directory (if running in RStudio)
#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# make directories for outputs
dir.create("tables", showWarnings = FALSE, recursive = TRUE)
dir.create("plots", showWarnings = FALSE, recursive = TRUE)

# Loading and processing data ####
## LMM data ####
gptNov <- read_csv("LMM_data/gpt4o_november_run.csv")
gpt4o <- read_csv("LMM_data/gpt4o_february_run.csv")
gpt4.1 <- read_csv("LMM_data/gpt4.1_may_run.csv")
gem1.5 <- read_csv("LMM_data/gemini1.5_november_run.csv")
gem2.5 <- read_csv("LMM_data/gemini2.5_may_run.csv")
llama4 <- read_csv("LMM_data/llama4_may_run.csv")
llama3.2 <- read_csv("LMM_data/llama3.2_may_run.csv")

## Survey data ####
pilotSubL<- read_dta('Survey_data/prolific_data.dta')
dmacsL<- read_csv('Survey_data/dmacs_data.csv')

# Tests ####
## Define functions ####
### T-Test functions ####
t_test_prolific <- function(model = c("GPT-4o","GPT-4.1","GPT-o4","Gemini-1.5","Gemini-2.5","Llama-4"),
                            LMM_data,
                            prolific_data = pilotSubL,
                            inner = F){
  #this function takes in the genAI model and survey data 
  #and outputs t-tests statistics comparing output from the model 
  #to output from the survey respondents for each image
  
  if(inner == F){
    message(paste("Running for model", model, "with Prolific using dataset", deparse(substitute(LMM_data)))) 
  }
  
  #create empty environments for each dataset
  prolific_na_env <- new.env()
  LMM_na_env <- new.env()
  
  ttestProlif <- lapply(unique(LMM_data$intro_bin), function(intro){
    #loop over prompt types
    genderout <- lapply(c(unique(prolific_data$gender), "both"), function(gender) {
      #loop over each gender and the full data
      out1 <- lapply(unique(prolific_data$img)[!is.na(unique(prolific_data$img))], function(image) {
        #loop over each image
        out <- lapply(unique(prolific_data$gptLabel), function(lab){
          #loop over each type of question
          
          if(gender == "both"){
            prolificRate <- prolific_data$value[prolific_data$img == image & prolific_data$gptLabel == lab]
            gptRate <- LMM_data$Response[LMM_data$`Image ID` == image & LMM_data$Question == lab & LMM_data$intro_bin == intro]
            
            # Collect NA information by image-question combo
            # For Prolific data
            if(any(is.na(prolificRate))){
              key <- paste(image, lab, sep = ", ")
              prolific_na_env[[key]] <- sum(is.na(prolificRate))
            }
            
            # For LMM data
            if(any(is.na(gptRate))){
              key <- paste(image, lab, sep = "_")
              LMM_na_env[[key]] <- sum(is.na(gptRate))
            }
          } else {
            prolificRate <- prolific_data$value[prolific_data$img == image & prolific_data$gptLabel == lab & prolific_data$gender == gender]
            gptRate <- LMM_data$Response[LMM_data$`Image ID` == image & LMM_data$Question == lab & LMM_data$intro_bin == intro]
          }
          #extracts out the correct subset for the image-intro-question-gender combo
          
          if (length(prolificRate) > 1 & length(gptRate) > 1) {
            #run a t.test and store the results
            ttest_result <- t.test(prolificRate, gptRate, na.rm = TRUE)
            data.frame(
              image_id = image,
              qType = lab,
              t_stat = ttest_result$statistic,
              p_value = ttest_result$p.value,
              intro = intro,
              gender = gender,
              meanSample = ttest_result$estimate[1],
              meanGPT = ttest_result$estimate[2],
              n_gpt = length(gptRate),
              n_sample=length(prolificRate),
              sample = "Prolific"
            )
          } else {
            warning(print(paste0(image, " with ", intro, " for ", gender, " and feature is ", lab, " has ", length(prolificRate) , " Prolific ratings and ",
                                 length(gptRate), " LMM ratings, so no t test was run.")))
            return(NULL)
          }
        })
        do.call("rbind", out)
      })
      do.call("rbind", out1)
    })
    do.call("rbind", genderout)
  })
  results <- do.call("rbind", ttestProlif)
  #combine all of the results 
  
  
  # Get the NA counts from the environments
  prolific_keys <- ls(prolific_na_env)
  if(length(prolific_keys) > 0){
    warning("\nSummary of NAs in Prolific data:")
    for(key in prolific_keys){
      parts <- strsplit(key, ", ")[[1]]
      image <- parts[1]
      question <- parts[2]
      warning(paste0(image, " has ", prolific_na_env[[key]], " NAs for ", question, " in prolific"))
    }
  } else {
    message("No NAs found in Prolific data")
  }
  
  LMM_keys <- ls(LMM_na_env)
  if(length(LMM_keys) > 0){
    warning("\nSummary of NAs in ", model, " data:")
    for(key in LMM_keys){
      parts <- strsplit(key, ", ")[[1]]
      image <- parts[1]
      question <- parts[2]
      warning(paste0(image, " has ", LMM_na_env[[key]], " NAs for ", question, " with ", model))
    }
  } else {
    message("No NAs found in ", model, " data")
  }
  
  return(results)
}


t_test_dmacs <- function(model = c("GPT-4o","GPT-4.1","GPT-o4","Gemini-1.5","Gemini-2.5","Llama-4"),
                         LMM_data,
                         dmacs_data = dmacsL,
                         inner = F){
  #completes the same model for the dmacs data with the correct weighted t-test functions
  
  if(inner == F){
    message(paste("Running for model", model, "with DMACS using dataset", deparse(substitute(LMM_data)))) 
  }
  
  dmacs_na_env <- new.env()
  LMM_na_env <- new.env()
  
  ttestDMACS <- lapply(unique(LMM_data$intro_bin), function(intro){
    genderout <- lapply(c(unique(dmacs_data$gender)[!is.na(unique(dmacs_data$gender))], "both"), function(gender) {
      out1 <- lapply(unique(dmacs_data$img)[!is.na(unique(dmacs_data$img))], function(image) {
        out <- lapply(unique(dmacs_data$gptLabel)[unique(dmacs_data$gptLabel) != "Income"], function(lab){
          
          if(gender == "both"){
            prolificRate <- dmacs_data$value[dmacs_data$img == image & dmacs_data$gptLabel == lab]
            weightS <- dmacs_data$weights[dmacs_data$img == image & dmacs_data$gptLabel == lab]
            gptRate <- LMM_data$Response[LMM_data$`Image ID` == image & LMM_data$Question == lab & LMM_data$intro_bin == intro]
            
            # Collect NA information by image-question combo
            # For Prolific data
            if(any(is.na(prolificRate))){
              key <- paste(image, lab, sep = ", ")
              dmacs_na_env[[key]] <- sum(is.na(prolificRate))
            }
            
            # For LMM data
            if(any(is.na(gptRate))){
              key <- paste(image, lab, sep = "_")
              LMM_na_env[[key]] <- sum(is.na(gptRate))
            }
            
          }else{
            prolificRate <- dmacs_data$value[dmacs_data$img == image & dmacs_data$gptLabel == lab & dmacs_data$gender == gender]
            weightS <- dmacs_data$weights[dmacs_data$img == image & dmacs_data$gptLabel == lab & dmacs_data$gender == gender]
            gptRate <- LMM_data$Response[LMM_data$`Image ID` == image & LMM_data$Question == lab & LMM_data$intro_bin == intro]
          }
          
          weightS <- weightS[!is.na(prolificRate)]
          prolificRate <- prolificRate[!is.na(prolificRate)]
          
          if(length(weightS) != length(prolificRate)){
            stop("There are a different amount of weights to ratings.")
          }
          
          if (length(prolificRate) > 1 & length(gptRate) > 1) {
            ttest_result <- weights::wtd.t.test(x = prolificRate, y = gptRate, weight = weightS, samedata = F,
                                                weighty = rep(1, length(gptRate)))
            data.frame(
              image_id = image,
              qType = lab,
              t_stat = as.numeric(unlist(ttest_result$coefficients["t.value"])),
              p_value = as.numeric(unlist(ttest_result$coefficients["p.value"])),
              intro = intro,
              gender = gender,
              meanSample = as.numeric(unlist(ttest_result$additional["Mean.x"])),
              meanGPT = as.numeric(unlist(ttest_result$additional["Mean.y"])),
              n_gpt = length(gptRate),
              n_sample=length(prolificRate),
              sample = "DMACS"
            )
          } else {
            warning(print(paste0(image, " with ", intro, " for ", gender, " and feature is ", lab, " has ", length(prolificRate) , " DMACS ratings and ",
                                 length(gptRate), " LMM ratings, so no t test was run.")))
            return(NULL)
          }
        })
        do.call("rbind", out)
      })
      do.call("rbind", out1)
    })
    do.call("rbind", genderout)
  })
  
  results <- do.call("rbind", ttestDMACS)
  
  
  # Get the NA counts from the environments
  prolific_keys <- ls(dmacs_na_env)
  if(length(prolific_keys) > 0){
    warning("\nSummary of NAs in DMACS data:")
    for(key in prolific_keys){
      parts <- strsplit(key, ", ")[[1]]
      image <- parts[1]
      question <- parts[2]
      warning(paste0(image, " has ", dmacs_na_env[[key]], " NAs for ", question, " in DMACS"))
    }
  } else {
    message("No NAs found in DMACS data")
  }
  
  LMM_keys <- ls(LMM_na_env)
  if(length(LMM_keys) > 0){
    warning("\nSummary of NAs in ", model, " data:")
    for(key in LMM_keys){
      parts <- strsplit(key, ", ")[[1]]
      image <- parts[1]
      question <- parts[2]
      warning(paste0(image, " has ", LMM_na_env[[key]], " NAs for ", question, " with ", model))
    }
  } else {
    message("No NAs found in ", model, " data")
  }
  
  return(results)
}

run_t_test <- function(model = c("GPT-4o","GPT-4.1","GPT-o4","Gemini-1.5","Gemini-2.5","Llama-4"),
                       LMM_data,
                       prolific_data = pilotSubL,
                       dmacs_data = dmacsL,
                       rename = T){
  #wrapper function that runs the t-tests functions for each of the datasets
  #combines them into a dataset that has plotting and table labels 
  
  message(paste("Running for model", model, "using dataset", deparse(substitute(LMM_data))))
  
  prolific <- t_test_prolific(model, LMM_data,prolific_data, inner= T)
  dmacs <- t_test_dmacs(model, LMM_data, dmacs_data, inner = T)
  
  t_test <- bind_rows(dmacs, prolific) %>% mutate(model = model)
  
  t_test$sig <- ifelse(t_test$p_value < 0.05,1, 0)
  
  out <- t_test %>% as.tibble() %>% 
    filter((sample == "DMACS" & gender == "both" & intro %in% c("Live in Detroit", "No prompt")) |
             (sample == "DMACS" & gender == "Female" & intro %in% c("Live in Detroit", "No prompt", "Woman", "Woman in Detroit")) |  
             (sample == "DMACS" & gender == "Male" & intro %in% c("Live in Detroit", "No prompt", "Man", "Man in Detroit")) | 
             (sample == "Prolific" & gender == "Female" & intro %in% c("No prompt", "Woman")) |
             (sample == "Prolific" & gender == "Male" & intro %in% c("No prompt", "Man")) |
             (sample == "Prolific" & gender == "both" & intro %in% c("No prompt"))  ) %>% 
    mutate(comparison = case_when(
      sample == "DMACS" & gender == "both" & intro %in% c("No prompt")  ~  paste("Detroit \n v \n", model),
      sample == "DMACS" & gender == "both" & intro %in% c("Live in Detroit") ~  paste("Detroit \n v \n", model, "('You live in Detroit')"),
      
      sample == "DMACS" & gender == "Female" & intro %in% c("No prompt") ~ paste("Detroit Women \n v \n", model),
      sample == "DMACS" & gender == "Female" & intro %in% c("Live in Detroit") ~ paste("Detroit Women \n v \n", model, "('You live in Detroit')"),
      sample == "DMACS" & gender == "Female" & intro %in% c( "Woman") ~ paste("Detroit Women \n v \n", model,"('You are a woman')"),
      sample == "DMACS" & gender == "Female" & intro %in% c("Woman in Detroit") ~ paste("Detroit Women \n v \n", model, "('You are a woman and live in Detroit')"),
      
      sample == "DMACS" & gender == "Male" & intro %in% c("No prompt") ~ paste("Detroit Men \n v \n", model),
      sample == "DMACS" & gender == "Male" & intro %in% c("Live in Detroit") ~ paste("Detroit Men \n v \n", model, "('You live in Detroit')"),
      sample == "DMACS" & gender == "Male" & intro %in% c( "Man") ~ paste("Detroit Men \n v \n", model, "('You are a man')"),
      sample == "DMACS" & gender == "Male" & intro %in% c("Man in Detroit") ~ paste("Detroit Men \n v \n", model, "('You are a man and live in Detroit')"),
      
      sample == "Prolific" & gender == "Female" & intro %in% c("No prompt") ~ paste("Women \n v \n", model), 
      sample == "Prolific" & gender == "Female" & intro %in% c("Woman") ~ paste("Women \n v \n", model, " ('You are a woman')"), 
      
      sample == "Prolific" & gender == "Male" & intro %in% c("No prompt") ~ paste("Men \n v \n", model), 
      sample == "Prolific" & gender == "Male" & intro %in% c("Man") ~ paste("Men \n v \n", model, " ('You are a man')"), 
      
      sample == "Prolific" & gender == "both" & intro %in% c("No prompt") ~ paste("Americans \n v \n", model)
    )) %>% 
    mutate(comparison_general = case_when(
      sample == "DMACS" & gender == "both" & intro %in% c("No prompt")  ~  paste("Detroit \n v \n", "LMM"),
      sample == "DMACS" & gender == "both" & intro %in% c("Live in Detroit") ~  paste("Detroit \n v \n", "LMM", "('You live in Detroit')"),
      
      sample == "DMACS" & gender == "Female" & intro %in% c("No prompt") ~ paste("Detroit Women \n v \n", "LMM"),
      sample == "DMACS" & gender == "Female" & intro %in% c("Live in Detroit") ~ paste("Detroit Women \n v \n", "LMM", "('You live in Detroit')"),
      sample == "DMACS" & gender == "Female" & intro %in% c( "Woman") ~ paste("Detroit Women \n v \n", "LMM","('You are a woman')"),
      sample == "DMACS" & gender == "Female" & intro %in% c("Woman in Detroit") ~ paste("Detroit Women \n v \n", "LMM", "('You are a woman and live in Detroit')"),
      
      sample == "DMACS" & gender == "Male" & intro %in% c("No prompt") ~ paste("Detroit Men \n v \n", "LMM"),
      sample == "DMACS" & gender == "Male" & intro %in% c("Live in Detroit") ~ paste("Detroit Men \n v \n", "LMM", "('You live in Detroit')"),
      sample == "DMACS" & gender == "Male" & intro %in% c( "Man") ~ paste("Detroit Men \n v \n", "LMM", "('You are a man')"),
      sample == "DMACS" & gender == "Male" & intro %in% c("Man in Detroit") ~ paste("Detroit Men \n v \n", "LMM", "('You are a man and live in Detroit')"),
      
      sample == "Prolific" & gender == "Female" & intro %in% c("No prompt") ~ paste("Women \n v \n", "LMM"), 
      sample == "Prolific" & gender == "Female" & intro %in% c("Woman") ~ paste("Women \n v \n", "LMM", "('You are a woman')"), 
      
      sample == "Prolific" & gender == "Male" & intro %in% c("No prompt") ~ paste("Men \n v \n", "LMM"), 
      sample == "Prolific" & gender == "Male" & intro %in% c("Man") ~ paste("Men \n v \n", "LMM", "('You are a man')"), 
      
      sample == "Prolific" & gender == "both" & intro %in% c("No prompt") ~ paste("Americans \n v \n", "LMM")
    ))
  return(out)
}

### Correlation functions ####
run_corr <- function(model = c("GPT-4o","GPT-4.1","GPT-o4","Gemini-1.5","Gemini-2.5","Llama-4"),
                     t_test_data){
  
  # creates correlation data from the survey averages extracted from the t-test tables and the LLM models
  
  out <- t_test_data %>% ungroup() %>% 
    group_by(comparison_general, qType) %>%
    summarise(
      correlation = cor(meanSample, meanGPT, use = "complete.obs"),
      n = sum(complete.cases(meanSample, meanGPT)),  
      corrSE = (1 - correlation^2) / sqrt(n - 2)
    ) %>%
    ungroup() %>% 
    mutate(model = model)
  
  return(out)
}

## Run tests ####
#provide labels for each model's associated dataset
model_data_mapping <- list(
  "GPT-4o" = "gpt4o",
  "GPT-4.1" = "gpt4.1",
  "Gemini-1.5" = "gem1.5",
  "Gemini-2.5" = "gem2.5",
  "Llama-4" = "llama4"
)

#set the models we want to test as the names of the genAI
models_to_test <- names(model_data_mapping)


# create an empty list to store results
t_test_list <- list()

# iterate through each model and run the t-test
for (model_name in models_to_test) {
  cat(paste("Processing model:", model_name, "\n"))
  
  dataset_name <- model_data_mapping[[model_name]]
  model_dataset <- get(dataset_name)
  
  # run t-test for specific model
  test_result <- run_t_test(model = model_name,
                            LMM_data = model_dataset,
                            prolific_data = pilotSubL,
                            dmacs_data = dmacsL)
  
  t_test_list[[model_name]] <- test_result
}

corr_list <- list()

# iterate through each model and run the correlation test
for (model_name in models_to_test) {
  cat(paste("Processing model:", model_name, "\n"))
  
  # run t-test for specific model
  test_result <- run_corr(model = model_name,
                          t_test_data = t_test_list[[model_name]])
  
  corr_list[[model_name]] <- test_result
}

# now we have corr_list and t_test_list

# Main paper plots ####
main_models <- bind_rows(t_test_list$`GPT-4o`,
                         t_test_list$`Gemini-2.5`,
                         t_test_list$`Llama-4`)

main_propsig<- main_models %>% dplyr::group_by(comparison_general, qType, model) %>% dplyr::summarize(
  propSigNeg = sum(sig*t_stat<0)/n(),
  propSigPos = sum(sig*t_stat>0)/n(), 
  propNonSig = 1 - mean(sig)) %>% 
  ungroup() %>%   filter(comparison_general %in% c("Detroit Women \n v \n LMM ('You are a woman and live in Detroit')",
                                                   "Detroit Women \n v \n LMM",
                                                   "Detroit Men \n v \n LMM ('You are a man and live in Detroit')",
                                                   "Detroit Men \n v \n LMM",
                                                   "Women \n v \n LMM ('You are a woman')",
                                                   "Women \n v \n LMM",
                                                   "Men \n v \n LMM ('You are a man')",
                                                   "Men \n v \n LMM",
                                                   "Detroit \n v \n LMM ('You live in Detroit')",
                                                   "Detroit \n v \n LMM",
                                                   "Americans \n v \n LMM")) %>%   
  pivot_wider(names_from = "model", values_from = 4:6) 

sum(main_propsig$`propNonSig_GPT-4o` > main_propsig$`propNonSig_Gemini-2.5`)/length(main_propsig$`propNonSig_Gemini-2.5`)
sum(main_propsig$`propNonSig_GPT-4o` > main_propsig$`propNonSig_Llama-4`)/length(main_propsig$`propNonSig_Llama-4`)

## Figure 1: GPT-4o, no genders ####
fig1p1<- t_test_list$`GPT-4o` %>% 
  filter(comparison_general %in% c("Americans \n v \n LMM",
                                   "Detroit \n v \n LMM",
                                   "Detroit \n v \n LMM ('You live in Detroit')")) %>% 
  ggplot(., aes(x = meanSample, y = meanGPT)) + 
  geom_point() + 
  facet_grid(comparison_general~factor(qType, levels = c("Wealth", "Safety - day", "Safety - night", "Disorder")), switch = "y")+
  ggtitle("")+
  labs(x = "Human Ratings",
       y = "GPT Ratings")+
  #scale_alpha_discrete(range = c(.3,1)) +
  scale_x_continuous(limits = c(0, 10), breaks = c(0,5,10))+
  scale_y_continuous(limits=c(0,10), breaks = c(0,5,10), position = "right")+
  geom_abline(intercept = 0, slope = 1, color = "firebrick3", linetype = "dashed") +
  theme_minimal() +
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "lm", se = TRUE, color = "skyblue4")+
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "loess", se = TRUE, color ="lightskyblue3" )+
  theme(panel.background = element_rect(fill = "white"),
        strip.placement = "outside",  # Ensure strip labels are outside
        strip.text.y.left = element_text(angle = 0),
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  # Add the correlation label at the top left of each facet
  geom_text(
    data = corr_list$`GPT-4o`  %>% 
      filter(comparison_general %in% c("Americans \n v \n LMM",
                                       "Detroit \n v \n LMM",
                                       "Detroit \n v \n LMM ('You live in Detroit')")),
    aes(
      x = 0.5, y = 9.5,  # Adjust the position as needed
      label = paste0("r = ", round(correlation, 2))
    ),
    color = "black",
    hjust = 0,
    size = 3
  ) 

fig1p2 <- main_propsig%>% 
  arrange(qType,comparison_general) %>% 
  select(comparison_general, qType, contains("GPT-4o")) %>% #MODEL HERE
  pivot_longer(cols = 3:5) %>% 
  mutate(nameLab = case_when(grepl("propSigNeg",name) ~  "GPT Significantly Over-Estimates", 
                             grepl("propSigPos",name) ~ "GPT Significantly Under-Estimates",
                             grepl("propNonSig",name) ~ "Non-Significant Difference"),
         qType = factor(qType, levels = c("Wealth", "Safety - day", "Safety - night", "Disorder"))) %>% 
  filter(comparison_general %in% c("Americans \n v \n LMM",
                                   "Detroit \n v \n LMM",
                                   "Detroit \n v \n LMM ('You live in Detroit')")) %>% 
  mutate(comparison_general = factor(comparison_general, levels = 
                                       c( "Detroit \n v \n LMM ('You live in Detroit')",
                                          "Detroit \n v \n LMM",
                                          "Americans \n v \n LMM"))) %>% 
  ggplot(., aes(x = value, y = comparison_general,  fill = nameLab)) + 
  geom_bar(stat = "identity") +
  facet_grid( ~qType, scales = "free_y") +
  labs(title = "",
       fill = "",
       x = "Proportion of Images",
       y = "") + 
  scale_fill_manual(values = c( "indianred4", "lightsalmon2", "bisque2")) + 
  theme_minimal() +
  theme(legend.position = "top",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks.x = element_line(),  # Ensure x-axis ticks are drawn
        axis.text.x = element_text())

everyone <-   fig1p1 + fig1p2 +
  plot_layout(ncol = 1, heights = c(2,1)) 

ggsave(filename = "plots/fig_1.png", everyone, width = 10, height = 9)

## Figure 2: GPT-4o, women ####
fig2<- t_test_list$`GPT-4o` %>% 
  filter(comparison_general %in% c("Women \n v \n LMM",
                                   "Women \n v \n LMM ('You are a woman')",
                                   "Detroit Women \n v \n LMM",
                                   "Detroit Women \n v \n LMM ('You are a woman and live in Detroit')")) %>% 
  mutate(comparison_general = factor(comparison_general, levels = c("Women \n v \n LMM",
                                                                    "Women \n v \n LMM ('You are a woman')",
                                                                    "Detroit Women \n v \n LMM",
                                                                    "Detroit Women \n v \n LMM ('You are a woman and live in Detroit')"))) %>%   
  ggplot(., aes(x = meanSample, y = meanGPT)) + 
  geom_point() + 
  facet_grid(comparison_general~factor(qType, levels = c("Wealth", "Safety - day", "Safety - night", "Disorder")), switch = "y")+
  ggtitle("")+
  labs(x = "Human Ratings",
       y = "GPT Ratings")+
  scale_x_continuous(limits = c(0, 10), breaks = c(0,5,10))+
  scale_y_continuous(limits=c(0,10), breaks = c(0,5,10), position = "right")+
  geom_abline(intercept = 0, slope = 1, color = "firebrick3", linetype = "dashed") +
  theme_minimal() +
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "lm", se = TRUE, color = "skyblue4")+
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "loess", se = TRUE, color ="lightskyblue3" )+
  theme(panel.background = element_rect(fill = "white"),
        strip.placement = "outside",  # Ensure strip labels are outside
        strip.text.y.left = element_text(angle = 0),
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  # Add the correlation label at the top left of each facet
  geom_text(
    data = corr_list$`GPT-4o`  %>%filter(comparison_general %in% c("Women \n v \n LMM",
                                                                   "Women \n v \n LMM ('You are a woman')",
                                                                   "Detroit Women \n v \n LMM",
                                                                   "Detroit Women \n v \n LMM ('You are a woman and live in Detroit')")) %>% 
      mutate(comparison_general = factor(comparison_general, levels = c("Women \n v \n LMM",
                                                                        "Women \n v \n LMM ('You are a woman')",
                                                                        "Detroit Women \n v \n LMM",
                                                                        "Detroit Women \n v \n LMM ('You are a woman and live in Detroit')"))),
    aes(
      x = 0.5, y = 9.5,  # Adjust the position as needed
      label = paste0("r = ", round(correlation, 2))
    ),
    color = "black",
    hjust = 0,
    size = 3
  ) 

ggsave(filename = "plots/fig2.png", fig2, width = 12, height = 10)

## Figure 3: GPT-4o, men ####
fig3<- t_test_list$`GPT-4o` %>% 
  filter(comparison_general %in% c("Men \n v \n LMM",
                                   "Men \n v \n LMM ('You are a man')",
                                   "Detroit Men \n v \n LMM",
                                   "Detroit Men \n v \n LMM ('You are a man and live in Detroit')")) %>% 
  mutate(comparison_general = factor(comparison_general, levels = c("Men \n v \n LMM",
                                                                    "Men \n v \n LMM ('You are a man')",
                                                                    "Detroit Men \n v \n LMM",
                                                                    "Detroit Men \n v \n LMM ('You are a man and live in Detroit')"))) %>%   
  ggplot(., aes(x = meanSample, y = meanGPT)) + 
  geom_point() + 
  facet_grid(comparison_general~factor(qType, levels = c("Wealth", "Safety - day", "Safety - night", "Disorder")), switch = "y")+
  ggtitle("")+
  labs(x = "Human Ratings",
       y = "GPT Ratings")+
  scale_x_continuous(limits = c(0, 10), breaks = c(0,5,10))+
  scale_y_continuous(limits=c(0,10), breaks = c(0,5,10), position = "right")+
  geom_abline(intercept = 0, slope = 1, color = "firebrick3", linetype = "dashed") +
  theme_minimal() +
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "lm", se = TRUE, color = "skyblue4")+
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "loess", se = TRUE, color ="lightskyblue3" )+
  theme(panel.background = element_rect(fill = "white"),
        strip.placement = "outside",  # Ensure strip labels are outside
        strip.text.y.left = element_text(angle = 0),
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  # Add the correlation label at the top left of each facet
  geom_text(
    data = corr_list$`GPT-4o`  %>%filter(comparison_general %in% c("Men \n v \n LMM",
                                                                   "Men \n v \n LMM ('You are a man')",
                                                                   "Detroit Men \n v \n LMM",
                                                                   "Detroit Men \n v \n LMM ('You are a man and live in Detroit')")) %>% 
      mutate(comparison_general = factor(comparison_general, levels = c("Men \n v \n LMM",
                                                                        "Men \n v \n LMM ('You are a man')",
                                                                        "Detroit Men \n v \n LMM",
                                                                        "Detroit Men \n v \n LMM ('You are a man and live in Detroit')"))),
    aes(
      x = 0.5, y = 9.5,  # Adjust the position as needed
      label = paste0("r = ", round(correlation, 2))
    ),
    color = "black",
    hjust = 0,
    size = 3
  ) 

ggsave(filename = "plots/fig3.png", fig3, width = 12, height = 10)

# Appendix plots ####
## Figure SI1: Gemini 2.5, all genders ####
# NB: the model needs to be specified in the geom_text call below as well as extracted from the t-test
figs1p1<- t_test_list$`Gemini-2.5` %>% 
  filter(comparison_general %in% c("Americans \n v \n LMM",
                                   "Detroit \n v \n LMM",
                                   "Detroit \n v \n LMM ('You live in Detroit')")) %>% 
  ggplot(., aes(x = meanSample, y = meanGPT)) + 
  geom_point() + 
  facet_grid(comparison_general~factor(qType, levels = c("Wealth", "Safety - day", "Safety - night", "Disorder")), switch = "y")+
  ggtitle("")+
  labs(x = "Human Ratings",
       y = "Gemini Ratings")+
  scale_x_continuous(limits = c(0, 10), breaks = c(0,5,10))+
  scale_y_continuous(limits=c(0,10), breaks = c(0,5,10), position = "right")+
  geom_abline(intercept = 0, slope = 1, color = "firebrick3", linetype = "dashed") +
  theme_minimal() +
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "lm", se = TRUE, color = "skyblue4")+
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "loess", se = TRUE, color ="lightskyblue3" )+
  theme(panel.background = element_rect(fill = "white"),
        strip.placement = "outside",  # Ensure strip labels are outside
        strip.text.y.left = element_text(angle = 0),
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  # Add the correlation label at the top left of each facet
  geom_text(
    data = corr_list$`Gemini-2.5`  %>% ### MODEL HERE TOO 
      filter(comparison_general %in% c("Americans \n v \n LMM",
                                       "Detroit \n v \n LMM",
                                       "Detroit \n v \n LMM ('You live in Detroit')")),
    aes(
      x = 0.5, y = 9.5, 
      label = paste0("r = ", round(correlation, 2))
    ),
    color = "black",
    hjust = 0,
    size = 3
  ) 

figs1p2 <- main_propsig%>% 
  arrange(qType,comparison_general) %>% 
  select(comparison_general, qType, contains("Gemini-2.5")) %>% #MODEL HERE
  pivot_longer(cols = 3:5) %>% 
  mutate(nameLab = case_when(grepl("propSigNeg",name) ~  "GPT Significantly Over-Estimates", 
                             grepl("propSigPos",name) ~ "GPT Significantly Under-Estimates",
                             grepl("propNonSig",name) ~ "Non-Significant Difference"),
         qType = factor(qType, levels = c("Wealth", "Safety - day", "Safety - night", "Disorder"))) %>% 
  filter(comparison_general %in% c("Americans \n v \n LMM",
                                   "Detroit \n v \n LMM",
                                   "Detroit \n v \n LMM ('You live in Detroit')")) %>% 
  mutate(comparison_general = factor(comparison_general, levels = 
                                       c( "Detroit \n v \n LMM ('You live in Detroit')",
                                          "Detroit \n v \n LMM",
                                          "Americans \n v \n LMM"))) %>% 
  ggplot(., aes(x = value, y = comparison_general,  fill = nameLab)) + 
  geom_bar(stat = "identity") +
  facet_grid( ~qType, scales = "free_y") +
  labs(title = "",
       fill = "",
       x = "Proportion of Images",
       y = "") + 
  scale_fill_manual(values = c( "indianred4", "lightsalmon2", "bisque2")) + 
  theme_minimal() +
  theme(legend.position = "top",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks.x = element_line(), 
        axis.text.x = element_text())

everyone <-   figs1p1 + figs1p2 +
  plot_layout(ncol = 1, heights = c(2,1)) 

ggsave(filename = "plots/fig_si1.png", everyone, width = 10, height = 9)

## Figure SI2: Gemini 2.5, men ####
si2<- t_test_list$`Gemini-2.5` %>% 
  filter(comparison_general %in% c("Men \n v \n LMM",
                                   "Men \n v \n LMM ('You are a man')",
                                   "Detroit Men \n v \n LMM",
                                   "Detroit Men \n v \n LMM ('You are a man and live in Detroit')")) %>% 
  mutate(comparison_general = factor(comparison_general, levels = c("Men \n v \n LMM",
                                                                    "Men \n v \n LMM ('You are a man')",
                                                                    "Detroit Men \n v \n LMM",
                                                                    "Detroit Men \n v \n LMM ('You are a man and live in Detroit')"))) %>%   
  ggplot(., aes(x = meanSample, y = meanGPT)) + 
  geom_point() + 
  facet_grid(comparison_general~factor(qType, levels = c("Wealth", "Safety - day", "Safety - night", "Disorder")), switch = "y")+
  ggtitle("")+
  labs(x = "Human Ratings",
       y = "Gemini Ratings")+
  scale_x_continuous(limits = c(0, 10), breaks = c(0,5,10))+
  scale_y_continuous(limits=c(0,10), breaks = c(0,5,10), position = "right")+
  geom_abline(intercept = 0, slope = 1, color = "firebrick3", linetype = "dashed") +
  theme_minimal() +
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "lm", se = TRUE, color = "skyblue4")+
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "loess", se = TRUE, color ="lightskyblue3" )+
  theme(panel.background = element_rect(fill = "white"),
        strip.placement = "outside",  
        strip.text.y.left = element_text(angle = 0),
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  # Add the correlation label at the top left of each facet
  geom_text(
    data = corr_list$`Gemini-2.5`  %>%filter(comparison_general %in% c("Men \n v \n LMM",
                                                                       "Men \n v \n LMM ('You are a man')",
                                                                       "Detroit Men \n v \n LMM",
                                                                       "Detroit Men \n v \n LMM ('You are a man and live in Detroit')")) %>% 
      mutate(comparison_general = factor(comparison_general, levels = c("Men \n v \n LMM",
                                                                        "Men \n v \n LMM ('You are a man')",
                                                                        "Detroit Men \n v \n LMM",
                                                                        "Detroit Men \n v \n LMM ('You are a man and live in Detroit')"))),
    aes(
      x = 0.5, y = 9.5, 
      label = paste0("r = ", round(correlation, 2))
    ),
    color = "black",
    hjust = 0,
    size = 3
  ) 

ggsave(filename = "plots/fig_si2.png", si2, width = 8, height = 6)


## Figure SI3: Gemini-2.5, women ####
si3<- t_test_list$`Gemini-2.5` %>% 
  filter(comparison_general %in% c("Women \n v \n LMM",
                                   "Women \n v \n LMM ('You are a woman')",
                                   "Detroit Women \n v \n LMM",
                                   "Detroit Women \n v \n LMM ('You are a woman and live in Detroit')")) %>% 
  mutate(comparison_general = factor(comparison_general, levels = c("Women \n v \n LMM",
                                                                    "Women \n v \n LMM ('You are a woman')",
                                                                    "Detroit Women \n v \n LMM",
                                                                    "Detroit Women \n v \n LMM ('You are a woman and live in Detroit')"))) %>%   
  ggplot(., aes(x = meanSample, y = meanGPT)) + 
  geom_point() + 
  facet_grid(comparison_general~factor(qType, levels = c("Wealth", "Safety - day", "Safety - night", "Disorder")), switch = "y")+
  ggtitle("")+
  labs(x = "Human Ratings",
       y = "Gemini Ratings")+
  scale_x_continuous(limits = c(0, 10), breaks = c(0,5,10))+
  scale_y_continuous(limits=c(0,10), breaks = c(0,5,10), position = "right")+
  geom_abline(intercept = 0, slope = 1, color = "firebrick3", linetype = "dashed") +
  theme_minimal() +
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "lm", se = TRUE, color = "skyblue4")+
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "loess", se = TRUE, color ="lightskyblue3" )+
  theme(panel.background = element_rect(fill = "white"),
        strip.placement = "outside",  # Ensure strip labels are outside
        strip.text.y.left = element_text(angle = 0),
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  # Add the correlation label at the top left of each facet
  geom_text(
    data = corr_list$`Gemini-2.5`  %>%filter(comparison_general %in% c("Women \n v \n LMM",
                                                                       "Women \n v \n LMM ('You are a woman')",
                                                                       "Detroit Women \n v \n LMM",
                                                                       "Detroit Women \n v \n LMM ('You are a woman and live in Detroit')")) %>% 
      mutate(comparison_general = factor(comparison_general, levels = c("Women \n v \n LMM",
                                                                        "Women \n v \n LMM ('You are a woman')",
                                                                        "Detroit Women \n v \n LMM",
                                                                        "Detroit Women \n v \n LMM ('You are a woman and live in Detroit')"))),
    aes(
      x = 0.5, y = 9.5, 
      label = paste0("r = ", round(correlation, 2))
    ),
    color = "black",
    hjust = 0,
    size = 3
  ) 

ggsave(filename = "plots/fig_si3.png", si3, width = 8, height = 6)


## Figure SI4: Llama-4, all genders ####

# NB: the model needs to be specified in the geom_text call below as well as extracted from the t-test
figs4p1<- t_test_list$`Llama-4` %>% 
  filter(comparison_general %in% c("Americans \n v \n LMM",
                                   "Detroit \n v \n LMM",
                                   "Detroit \n v \n LMM ('You live in Detroit')")) %>% 
  ggplot(., aes(x = meanSample, y = meanGPT)) + 
  geom_point() + 
  facet_grid(comparison_general~factor(qType, levels = c("Wealth", "Safety - day", "Safety - night", "Disorder")), switch = "y")+
  ggtitle("")+
  labs(x = "Human Ratings",
       y = "Llama Ratings")+
  scale_x_continuous(limits = c(0, 10), breaks = c(0,5,10))+
  scale_y_continuous(limits=c(0,10), breaks = c(0,5,10), position = "right")+
  geom_abline(intercept = 0, slope = 1, color = "firebrick3", linetype = "dashed") +
  theme_minimal() +
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "lm", se = TRUE, color = "skyblue4")+
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "loess", se = TRUE, color ="lightskyblue3" )+
  theme(panel.background = element_rect(fill = "white"),
        strip.placement = "outside", 
        strip.text.y.left = element_text(angle = 0),
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  # Add the correlation label at the top left of each facet
  geom_text(
    data = corr_list$`Llama-4`  %>% ### MODEL HERE TOO 
      filter(comparison_general %in% c("Americans \n v \n LMM",
                                       "Detroit \n v \n LMM",
                                       "Detroit \n v \n LMM ('You live in Detroit')")),
    aes(
      x = 0.5, y = 9.5,  
      label = paste0("r = ", round(correlation, 2))
    ),
    color = "black",
    hjust = 0,
    size = 3
  ) 

figs4p2 <- main_propsig%>% 
  arrange(qType,comparison_general) %>% 
  select(comparison_general, qType, contains("Llama-4")) %>% #MODEL HERE
  pivot_longer(cols = 3:5) %>% 
  mutate(nameLab = case_when(grepl("propSigNeg",name) ~  "GPT Significantly Over-Estimates", 
                             grepl("propSigPos",name) ~ "GPT Significantly Under-Estimates",
                             grepl("propNonSig",name) ~ "Non-Significant Difference"),
         qType = factor(qType, levels = c("Wealth", "Safety - day", "Safety - night", "Disorder"))) %>% 
  filter(comparison_general %in% c("Americans \n v \n LMM",
                                   "Detroit \n v \n LMM",
                                   "Detroit \n v \n LMM ('You live in Detroit')")) %>% 
  mutate(comparison_general = factor(comparison_general, levels = 
                                       c( "Detroit \n v \n LMM ('You live in Detroit')",
                                          "Detroit \n v \n LMM",
                                          "Americans \n v \n LMM"))) %>% 
  ggplot(., aes(x = value, y = comparison_general,  fill = nameLab)) + 
  geom_bar(stat = "identity") +
  facet_grid( ~qType, scales = "free_y") +
  labs(title = "",
       fill = "",
       x = "Proportion of Images",
       y = "") + 
  scale_fill_manual(values = c( "indianred4", "lightsalmon2", "bisque2")) + 
  theme_minimal() +
  theme(legend.position = "top",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks.x = element_line(),  
        axis.text.x = element_text())

everyone <-   figs4p1 + figs4p2 +
  plot_layout(ncol = 1, heights = c(2,1)) 

ggsave(filename = "plots/fig_si4.png", everyone, width = 10, height = 9)

## Figure SI5: Llama-4, men ####
si5<- t_test_list$`Llama-4` %>% 
  filter(comparison_general %in% c("Men \n v \n LMM",
                                   "Men \n v \n LMM ('You are a man')",
                                   "Detroit Men \n v \n LMM",
                                   "Detroit Men \n v \n LMM ('You are a man and live in Detroit')")) %>% 
  mutate(comparison_general = factor(comparison_general, levels = c("Men \n v \n LMM",
                                                                    "Men \n v \n LMM ('You are a man')",
                                                                    "Detroit Men \n v \n LMM",
                                                                    "Detroit Men \n v \n LMM ('You are a man and live in Detroit')"))) %>%   
  ggplot(., aes(x = meanSample, y = meanGPT)) + 
  geom_point() + 
  facet_grid(comparison_general~factor(qType, levels = c("Wealth", "Safety - day", "Safety - night", "Disorder")), switch = "y")+
  ggtitle("")+
  labs(x = "Human Ratings",
       y = "Llama Ratings")+
  scale_x_continuous(limits = c(0, 10), breaks = c(0,5,10))+
  scale_y_continuous(limits=c(0,10), breaks = c(0,5,10), position = "right")+
  geom_abline(intercept = 0, slope = 1, color = "firebrick3", linetype = "dashed") +
  theme_minimal() +
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "lm", se = TRUE, color = "skyblue4")+
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "loess", se = TRUE, color ="lightskyblue3" )+
  theme(panel.background = element_rect(fill = "white"),
        strip.placement = "outside",  # Ensure strip labels are outside
        strip.text.y.left = element_text(angle = 0),
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  # Add the correlation label at the top left of each facet
  geom_text(
    data = corr_list$`Llama-4`  %>%
      filter(comparison_general %in% c("Men \n v \n LMM",
                                       "Men \n v \n LMM ('You are a man')",
                                       "Detroit Men \n v \n LMM",
                                       "Detroit Men \n v \n LMM ('You are a man and live in Detroit')")) %>% 
      mutate(comparison_general = factor(comparison_general, levels = c("Men \n v \n LMM",
                                                                        "Men \n v \n LMM ('You are a man')",
                                                                        "Detroit Men \n v \n LMM",
                                                                        "Detroit Men \n v \n LMM ('You are a man and live in Detroit')"))),
    aes(
      x = 0.5, y = 9.5,  
      label = paste0("r = ", round(correlation, 2))
    ),
    color = "black",
    hjust = 0,
    size = 3
  ) 

ggsave(filename = "plots/fig_si5.png", si5, width = 8, height = 6)


## Figure SI6: Llama-4, women ####
si6<- t_test_list$`Llama-4` %>% 
  filter(comparison_general %in% c("Women \n v \n LMM",
                                   "Women \n v \n LMM ('You are a woman')",
                                   "Detroit Women \n v \n LMM",
                                   "Detroit Women \n v \n LMM ('You are a woman and live in Detroit')")) %>% 
  mutate(comparison_general = factor(comparison_general, levels = c("Women \n v \n LMM",
                                                                    "Women \n v \n LMM ('You are a woman')",
                                                                    "Detroit Women \n v \n LMM",
                                                                    "Detroit Women \n v \n LMM ('You are a woman and live in Detroit')"))) %>%   
  ggplot(., aes(x = meanSample, y = meanGPT)) + 
  geom_point() + 
  facet_grid(comparison_general~factor(qType, levels = c("Wealth", "Safety - day", "Safety - night", "Disorder")), switch = "y")+
  ggtitle("")+
  labs(x = "Human Ratings",
       y = "Llama Ratings")+
  scale_x_continuous(limits = c(0, 10), breaks = c(0,5,10))+
  scale_y_continuous(limits=c(0,10), breaks = c(0,5,10), position = "right")+
  geom_abline(intercept = 0, slope = 1, color = "firebrick3", linetype = "dashed") +
  theme_minimal() +
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "lm", se = TRUE, color = "skyblue4")+
  geom_smooth(aes(x = meanSample, y = meanGPT), method = "loess", se = TRUE, color ="lightskyblue3" )+
  theme(panel.background = element_rect(fill = "white"),
        strip.placement = "outside",  
        strip.text.y.left = element_text(angle = 0),
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  # Add the correlation label at the top left of each facet
  geom_text(
    data = corr_list$`Llama-4`  %>%filter(comparison_general %in% c("Women \n v \n LMM",
                                                                    "Women \n v \n LMM ('You are a woman')",
                                                                    "Detroit Women \n v \n LMM",
                                                                    "Detroit Women \n v \n LMM ('You are a woman and live in Detroit')")) %>% 
      mutate(comparison_general = factor(comparison_general, levels = c("Women \n v \n LMM",
                                                                        "Women \n v \n LMM ('You are a woman')",
                                                                        "Detroit Women \n v \n LMM",
                                                                        "Detroit Women \n v \n LMM ('You are a woman and live in Detroit')"))),
    aes(
      x = 0.5, y = 9.5,  # Adjust the position as needed
      label = paste0("r = ", round(correlation, 2))
    ),
    color = "black",
    hjust = 0,
    size = 3
  ) 

ggsave(filename = "plots/fig_si6.png", si6, width = 8, height = 6)



## Figure SI7: Mean Correlations ####
# weighted mean correlations per model
weighted_mean_corr <- function(model_data,model_name) {
  weights <- model_data$n - 3
  
  # Fisher's z transformation for each correlation
  z_values <- atanh(model_data$correlation)
  
  # weighted mean in z-space
  weighted_mean_z <- sum(z_values * weights) / sum(weights)
  
  # standard error for weighted mean
  se_z <- sqrt(1 / sum(weights))
  
  ci_lower_z <- weighted_mean_z - 1.96 * se_z
  ci_upper_z <- weighted_mean_z + 1.96 * se_z
  
  # convert back to correlation scale
  mean_r <- tanh(weighted_mean_z)
  ci_lower_r <- tanh(ci_lower_z)
  ci_upper_r <- tanh(ci_upper_z)
  
  return(data.frame(
    mean_correlation = mean_r,
    ci_lower = ci_lower_r,
    ci_upper = ci_upper_r,
    total_n = sum(model_data$n), 
    model_name = model_name
  ))
}

# Apply to each model's data
model_results <- rbind(weighted_mean_corr(corr_list$`GPT-4o`,"GPT-4o"),
                       weighted_mean_corr(corr_list$`Gemini-1.5`,"Gemini 1.5 Pro"),
                       weighted_mean_corr(corr_list$`Llama-4`, "Llama 4"),
                       weighted_mean_corr(corr_list$`GPT-4.1`, "GPT 4.1"),
                       weighted_mean_corr(corr_list$`Gemini-2.5`, "Gemini 2.5 Pro")
)

figsi7<- ggplot(model_results, aes(x = reorder(model_name, -mean_correlation), y = mean_correlation)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), 
                width = 0.2, linewidth = 1) +
  labs(
    x = NULL,
    y = "Mean Correlation (Fisher z-transformed)"
  ) +
  scale_y_continuous(limits = c(0.3, 0.7), breaks = seq(0.3, 0.7, 0.1)) +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.background = element_rect(fill = "white", color = "white"),
        panel.background = element_rect(fill = "white", color = "white"))

ggsave("plots/fig_si7.png",figsi7, width=8,height=4)



## Figure SI8: Labeler Power Analysis ####
df_power<-read_csv("Survey_data/pilot_data_power_analysis.csv")

df_f <- df_power %>%
  group_by(varLabel) %>%
  mutate(
    value = (value - min(value,na.rm = T)) / (max(value,na.rm = T) - min(value,na.rm = T))
  ) %>%
  ungroup() 

num_simulations <- 1000  

# Function to simulate for a single n
simulate_for_n <- function(n, column) {
  sampled_means <- replicate(num_simulations, {
    sampled_data <- column[sample(length(column), n, replace = TRUE)]
    mean(sampled_data,na.rm=T)
  })
  
  return(sampled_means)
}

out <- data.frame(n = integer(), variance = numeric(), Q = character())
n_values <- seq(2, 50, by = 2)
question_list <- c("Wealth", "Disorder", "Daytime Walking", "Nighttime Walking")

for (question in question_list) {
  df_x <- df_f %>% filter(grepl(question, varLabel))
  
  results <- data.frame(n = integer(), variance = numeric())
  
  # Simulation loop over n values
  for (n in n_values) {
    sim_results <- df_x %>%
      group_by(image_id) %>%
      summarise(
        sampled_means = list(simulate_for_n(n, value))
      ) %>%
      unnest(col = sampled_means) %>%
      summarise(
        variance = var(sampled_means, na.rm = TRUE)
      )
    
    # Store results for the current n
    results <- rbind(results, data.frame(n = n, variance = sim_results$variance))
  }
  
  print(paste0("Question ", question, " done"))
  
  out <- rbind(out, results %>% mutate(Question = question))
}


variance <- ggplot(out, aes(x = n, y = variance, color=Question)) +
  geom_line(linewidth=1) +
  labs(title = "Effect of Increasing n on variance of sample means",
       x = "Number of Labels per Image (n)",
       y = "Variance of Sample Means") +
  theme_minimal()

ggsave("plots/fig_si8.png",variance,width=7,height=5)


## Figure SI9: Detroit sampling map ####
dmacs_coords<-read_csv("LMM_data/images_with_coordinates.csv")

# Get the map data of Detroit from OpenStreetMap
detroit_bbox <- getbb("detroit, Michigan")
detroit_map <- opq(bbox = detroit_bbox) %>%
  add_osm_feature(key = "highway", value = "residential") %>%
  osmdata_sf()

# Plot 85 coordinates from study 1
y<-ggplot() +
  geom_sf(data = detroit_map$osm_lines, color = "gray", size = 0.1)+
  geom_point(data = dmacs_coords, aes(x = longitude, y = latitude), color = "red", size=3, alpha = 1) +
  coord_sf(crs = 4326) +
  labs(title = "Map of scraped images in Detroit",x="",y="")

ggsave("plots/fig_si9.png", width=6,height=6)

## Figure SI11: LMMs over time ####
### GPT nov vs now 
compare_gptNov_gpt <- gptNov  %>% 
  group_by(Question, intro, `Image ID`) %>%
  summarise(mean_response= mean(Response,na.rm=T)) %>% 
  rename(mean_nov = mean_response) %>%
  left_join(gpt4o %>%
              group_by(Question, intro, `Image ID`) %>%
              summarise(mean_response= mean(Response,na.rm=T)) %>% 
              rename(mean_later = mean_response),
            by = c("Question", "intro", "Image ID")
  ) %>% 
  mutate(diff = mean_nov - mean_later) 

compare_gptNov_gpt_correlation <-
  compare_gptNov_gpt %>% 
  mutate(intro = case_when(intro == "You are a man living in Detroit"~"You are a man\nliving in Detroit",
                           intro == "You are a woman living in Detroit"~"You are a woman\nliving in Detroit",
                           TRUE ~ intro)) %>% 
  summarise(correlation = cor(mean_nov, mean_later, use = "complete.obs")) 


compare_gptNov_gpt_plot<-
  compare_gptNov_gpt %>% 
  mutate(intro = case_when(intro == "You are a man living in Detroit"~"You are a man\nliving in Detroit",
                           intro == "You are a woman living in Detroit"~"You are a woman\nliving in Detroit",
                           TRUE ~ intro)) %>% 
  ungroup %>% 
  ggplot(., aes(y = mean_nov, x = mean_later)) + 
  geom_point() +
  facet_grid(intro~Question,switch = "y")+
  ggtitle("")+
  labs(x = "GPT ratings in February",
       y = "GPT ratings in November")+
  scale_x_continuous(limits = c(0, 10), breaks = c(0,5,10))+
  scale_y_continuous(limits=c(0,10), breaks = c(0,5,10), position = "right")+
  geom_abline(intercept = 0, slope = 1, color = "firebrick3", linetype = "dashed") +
  theme_minimal() +
  geom_smooth(aes(x = mean_later, y = mean_nov), method = "lm", se = TRUE, color = "skyblue4")+
  theme(panel.background = element_rect(fill = "white"),
        strip.placement = "outside",  # Ensure strip labels are outside
        strip.text.y.left = element_text(angle = 0),
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())+ 
  geom_text(
    data = compare_gptNov_gpt_correlation,
    aes(
      x = 0.5, y = 8, 
      label = paste0("r = ", round(correlation, 2))
    ),
    color = "black",
    hjust = 0,
    size = 3
  ) 

### Gemini nov vs now 
gem_repeat <- read_csv("LMM_data/gemini1.5_repeat_test.csv") %>% 
  mutate(Question = case_when(grepl("how safe would you feel walking around this neighborhood after dark",question_text)~ "Safety - night",
                              grepl("daylight",question_text)~ "Safety - day",
                              grepl("disorderly",question_text)~ "Disorder",
                              grepl("wealthy",question_text)~ "Wealth")) %>% 
  mutate(Refusal = ifelse(grepl("sorry",Response),1,0)) %>% 
  filter(Response != "Error") %>%
  mutate(Response = ifelse(Response == "", NA, Response)) %>% 
  mutate(respond = ifelse(grepl("respond",question_text,ignore.case = T),1,0),
         man = ifelse(grepl("a man",question_text,ignore.case = T),1,0),
         woman = ifelse(grepl("woman",question_text,ignore.case = T),1,0),
         detroit = ifelse(grepl("Detroit",question_text,ignore.case = T),1,0),
         plain = ifelse(man + woman + detroit ==0, 1,0)) %>% 
  mutate(intro = ifelse(plain==0,
                        sub("\\..*", "", question_text),"No prompt")) %>% 
  mutate(intro_bin = case_when(man == 1 & detroit ==1 ~ "Man in Detroit",
                               woman == 1 & detroit == 1 ~"Woman in Detroit",
                               man == 1 ~ "Man",
                               woman == 1 ~ "Woman",
                               man ==0 & woman == 0 & detroit ==1 ~ "Live in Detroit",
                               intro == "No prompt" ~ "No prompt"))

compare_gemNov_gem <- gem1.5 %>% 
  group_by(Question, intro, `Image ID`) %>%
  summarise(mean_response= mean(Response,na.rm=T)) %>% 
  rename(mean_nov = mean_response) %>%
  left_join(gem_repeat %>%
              group_by(Question, intro, `Image ID`) %>%
              summarise(mean_response= mean(Response,na.rm=T)) %>% 
              rename(mean_later = mean_response),
            by = c("Question", "intro", "Image ID")
  ) %>% 
  mutate(diff = mean_nov - mean_later) 

compare_gemNov_gem_correlation <-
  compare_gemNov_gem %>% 
  mutate(intro = case_when(intro == "You are a man living in Detroit"~"You are a man\nliving in Detroit",
                           intro == "You are a woman living in Detroit"~"You are a woman\nliving in Detroit",
                           TRUE ~ intro)) %>% 
  summarise(correlation = cor(mean_nov, mean_later, use = "complete.obs")) 

compare_gemNov_gem_plot<-
  compare_gemNov_gem %>% 
  mutate(intro = case_when(intro == "You are a man living in Detroit"~"You are a man\nliving in Detroit",
                           intro == "You are a woman living in Detroit"~"You are a woman\nliving in Detroit",
                           TRUE ~ intro)) %>% 
  ungroup %>% 
  ggplot(., aes(y = mean_nov, x = mean_later)) + 
  geom_point() +
  facet_grid(intro~Question,switch = "y")+
  ggtitle("")+
  labs(x = "Gemini 1.5 Pro ratings in May",
       y = "Gemini 1.5 Pro ratings in November")+
  scale_x_continuous(limits = c(0, 10), breaks = c(0,5,10))+
  scale_y_continuous(limits=c(0,10), breaks = c(0,5,10), position = "right")+
  geom_abline(intercept = 0, slope = 1, color = "firebrick3", linetype = "dashed") +
  theme_minimal() +
  geom_smooth(aes(x = mean_later, y = mean_nov), method = "lm", se = TRUE, color = "skyblue4")+
  theme(panel.background = element_rect(fill = "white"),
        strip.placement = "outside",  # Ensure strip labels are outside
        strip.text.y.left = element_text(angle = 0),
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())+ 
  geom_text(
    data = compare_gemNov_gem_correlation,
    aes(
      x = 0.5, y = 8, 
      label = paste0("r = ", round(correlation, 2))
    ),
    color = "black",
    hjust = 0,
    size = 3
  ) 

over_time<- compare_gptNov_gpt_plot + compare_gemNov_gem_plot +
  plot_layout(ncol = 1, heights = c(1,1)) 

ggsave(filename = "plots/fig_si11.png", over_time, width = 7, height = 10)

# Appendix tables ####
## Table SI1: Summary of all results ####
main_correlations <- 
  bind_rows(corr_list$`GPT-4o`, corr_list$`Gemini-2.5`, corr_list$`Llama-4`) %>% 
  filter(comparison_general %in% c("Detroit Women \n v \n LMM ('You are a woman and live in Detroit')",
                                   "Detroit Women \n v \n LMM",
                                   "Detroit Men \n v \n LMM ('You are a man and live in Detroit')",
                                   "Detroit Men \n v \n LMM",
                                   "Women \n v \n LMM ('You are a woman')",
                                   "Women \n v \n LMM",
                                   "Men \n v \n LMM ('You are a man')",
                                   "Men \n v \n LMM",
                                   "Detroit \n v \n LMM ('You live in Detroit')",
                                   "Detroit \n v \n LMM",
                                   "Americans \n v \n LMM")) %>%   
  pivot_wider(names_from = "model", values_from = c("correlation","corrSE")) %>% select(-n)

# get performance comparisons: 
sum(main_correlations$`correlation_GPT-4o` > main_correlations$`correlation_Gemini-2.5`)/length(main_correlations$`correlation_Gemini-2.5`)
sum(main_correlations$`correlation_GPT-4o` > main_correlations$`correlation_Llama-4`)/length(main_correlations$`correlation_Llama-4`)

### table out####
table_si1<- left_join(main_correlations, main_propsig, by = c("comparison_general", "qType"))%>% 
  arrange(qType,comparison_general) %>% 
  mutate_at(vars(3:17), round, digits = 2) %>% 
  relocate(starts_with("propNonSig"), .before = "propSigNeg_GPT-4o") %>% 
  kable(format = "latex", booktabs = TRUE, row.names = FALSE, escape = FALSE) %>%
  kable_styling(latex_options = c("hold_position")) %>%
  add_header_above(c(" " = 2, "GPT" = 1, "Gemini" = 1, "Llama" = 1,
                     "GPT" = 1, "Gemini" = 1, "Llama" = 1,
                     "GPT" = 1, "Gemini" = 1, "Llama" = 1,
                     "GPT" = 1, "Gemini" = 1,"Llama" = 1,
                     "GPT" = 1, "Gemini" = 1, "Llama" = 1)) %>% 
  add_header_above(c("Comparison" = 1, "Attribute" = 1, "Correlation" = 3, "Correlation SE" = 3,
                     "Prop non Significant" = 3, "Prop genai over" = 3,"Prop genai under" = 3)) 
writeLines(as.character(table_si1), "tables/table_si1.tex")

## Tables SI2-SI4: GPT-4o, Gemini2.5, Llama summary stats ####

#get summary statistics of the models answers for each question and model
models_summary <- main_models %>% 
  group_by(qType, comparison_general,model) %>% 
  dplyr::summarize(Mean = mean(meanGPT, na.rm = TRUE),
                   Median = median(meanGPT, na.rm = TRUE),
                   sd = sd(meanGPT, na.rm = TRUE),
                   min = min(meanGPT, na.rm = TRUE),
                   max = max(meanGPT, na.rm = TRUE)) %>% 
  mutate(comparison_general = str_remove(comparison_general,"\n")) %>% 
  mutate(across(where(is.numeric), round, digits = 2)) 

#get summary statistics for respondents' answers to each question
sample_summary <- main_models %>% group_by(qType, comparison_general) %>% 
  dplyr::summarize(Mean = mean(meanSample, na.rm = TRUE),
                   Median = median(meanSample, na.rm = TRUE),
                   sd = sd(meanSample, na.rm = TRUE),
                   min = min(meanSample, na.rm = TRUE),
                   max = max(meanSample, na.rm = TRUE)) %>% 
  mutate(model = "Sample") %>% 
  mutate(comparison_general = str_remove(comparison_general,"\n")) %>% 
  mutate(across(where(is.numeric), round, digits = 2)) 

#combine into one table 
all_summary <- bind_rows(models_summary,sample_summary) %>% 
  pivot_wider(names_from = "model",
              values_from = 4:8) %>% 
  relocate(comparison_general, .before = qType) %>% 
  filter(!(comparison_general %in% c("Detroit Women  v \n LMM ('You live in Detroit')",
                                     "Detroit Men  v \n LMM ('You live in Detroit')",
                                     "Detroit Men  v \n LMM ('You are a man')",
                                     "Detroit Women  v \n LMM ('You are a woman')")))

### Table SI2 out: both genders ####
table_si2<- all_summary %>% filter(!grepl("Women", comparison_general), !grepl("Men", comparison_general)) %>% 
  arrange(qType,comparison_general) %>% 
  kable(format = "latex", booktabs = TRUE, row.names = FALSE, escape = FALSE) %>%
  kable_styling(latex_options = c("hold_position")) %>%
  add_header_above(c(" " = 2, "GPT" = 1, "Gemini" = 1, "Llama" = 1,  "Sample" = 1, 
                     "GPT" = 1, "Gemini" = 1,"Llama" = 1, "Sample" = 1, 
                     "GPT" = 1, "Gemini" = 1, "Llama" = 1, "Sample" = 1, 
                     "GPT" = 1, "Gemini" = 1, "Llama" = 1, "Sample" = 1, 
                     "GPT" = 1,"Gemini" = 1, "Llama" = 1, "Sample" = 1)) %>% 
  add_header_above(c("Comparison" = 1,"Question" = 1, "Mean" = 4, "Median" = 4, "SD" = 4, "Min" = 4, "Max" =4)) 
writeLines(as.character(table_si2), "tables/table_si2.tex")

### Table SI3 out: women ####
table_si3<- all_summary %>% filter(grepl("Women", comparison_general))%>% 
  arrange(qType,comparison_general) %>% 
  kable(format = "latex", booktabs = TRUE, row.names = FALSE, escape = FALSE) %>%
  kable_styling(latex_options = c("hold_position")) %>%
  add_header_above(c(" " = 2, "GPT" = 1, "Gemini" = 1, "Llama" = 1,  "Sample" = 1, 
                     "GPT" = 1, "Gemini" = 1,"Llama" = 1, "Sample" = 1, 
                     "GPT" = 1, "Gemini" = 1, "Llama" = 1, "Sample" = 1, 
                     "GPT" = 1, "Gemini" = 1, "Llama" = 1, "Sample" = 1, 
                     "GPT" = 1,"Gemini" = 1, "Llama" = 1, "Sample" = 1)) %>% 
  add_header_above(c("Comparison" = 1,"Question" = 1, "Mean" = 4, "Median" = 4, "SD" = 4, "Min" = 4, "Max" =4)) 
writeLines(as.character(table_si3), "tables/table_si3.tex")

### Table SI4 out: men ####
table_si4<- all_summary %>% filter(grepl("Men", comparison_general)) %>% 
  arrange(qType,comparison_general) %>% 
  kable(format = "latex", booktabs = TRUE, row.names = FALSE, escape = FALSE) %>%
  kable_styling(latex_options = c("hold_position")) %>%
  add_header_above(c(" " = 2, "GPT" = 1, "Gemini" = 1, "Llama" = 1,  "Sample" = 1, 
                     "GPT" = 1, "Gemini" = 1,"Llama" = 1, "Sample" = 1, 
                     "GPT" = 1, "Gemini" = 1, "Llama" = 1, "Sample" = 1, 
                     "GPT" = 1, "Gemini" = 1, "Llama" = 1, "Sample" = 1, 
                     "GPT" = 1,"Gemini" = 1, "Llama" = 1, "Sample" = 1)) %>% 
  add_header_above(c("Comparison" = 1,"Question" = 1, "Mean" = 4, "Median" = 4, "SD" = 4, "Min" = 4, "Max" =4)) 
writeLines(as.character(table_si4), "tables/table_si4.tex")

## Table SI5: Summary of Results for Alternative Models ####
secondary_models <- bind_rows(t_test_list$`GPT-4.1`,
                              t_test_list$`Gemini-2.5`)


secondary_propsig<- secondary_models %>% dplyr::group_by(comparison_general, qType, model) %>% dplyr::summarize(
  propSigNeg = sum(sig*t_stat<0)/n(),
  propSigPos = sum(sig*t_stat>0)/n(), 
  propNonSig = 1 - mean(sig)) %>% 
  ungroup() %>%   filter(comparison_general %in% c("Detroit Women \n v \n LMM ('You are a woman and live in Detroit')",
                                                   "Detroit Women \n v \n LMM",
                                                   "Detroit Men \n v \n LMM ('You are a man and live in Detroit')",
                                                   "Detroit Men \n v \n LMM",
                                                   "Women \n v \n LMM ('You are a woman')",
                                                   "Women \n v \n LMM",
                                                   "Men \n v \n LMM ('You are a man')",
                                                   "Men \n v \n LMM",
                                                   "Detroit \n v \n LMM ('You live in Detroit')",
                                                   "Detroit \n v \n LMM",
                                                   "Americans \n v \n LMM")) %>%   
  pivot_wider(names_from = "model", values_from = 4:6) 

secondary_correlations <- 
  bind_rows(corr_list$`GPT-4.1`, corr_list$`Gemini-1.5`) %>% 
  filter(comparison_general %in% c("Detroit Women \n v \n LMM ('You are a woman and live in Detroit')",
                                   "Detroit Women \n v \n LMM",
                                   "Detroit Men \n v \n LMM ('You are a man and live in Detroit')",
                                   "Detroit Men \n v \n LMM",
                                   "Women \n v \n LMM ('You are a woman')",
                                   "Women \n v \n LMM",
                                   "Men \n v \n LMM ('You are a man')",
                                   "Men \n v \n LMM",
                                   "Detroit \n v \n LMM ('You live in Detroit')",
                                   "Detroit \n v \n LMM",
                                   "Americans \n v \n LMM")) %>%   
  pivot_wider(names_from = "model", values_from = c("correlation","corrSE")) %>% select(-n)

### table out####
table_si5<- left_join(secondary_correlations, secondary_propsig, by = c("comparison_general", "qType"))%>% 
  arrange(qType,comparison_general) %>% 
  mutate_at(vars(3:12), round, digits = 2) %>% 
  relocate(starts_with("propNonSig"), .before = "propSigNeg_GPT-4.1") %>% 
  kable(format = "latex", booktabs = TRUE, row.names = FALSE, escape = FALSE) %>%
  kable_styling(latex_options = c("hold_position")) %>%
  add_header_above(c(" " = 2, "GPT 4.1" = 1, "Gemini 2.5" = 1, 
                     "GPT 4.1" = 1, "Gemini 2.5" = 1, 
                     "GPT 4.1" = 1, "Gemini 2.5" = 1, 
                     "GPT 4.1" = 1, "Gemini 2.5" = 1, 
                     "GPT 4.1" = 1, "Gemini 2.5" = 1 )) %>% 
  add_header_above(c("Comparison" = 1, "Attribute" = 1, "Correlation" = 2, "Correlation SE" = 2,
                     "Prop Non-Significant" = 2, "Prop GenAI Sig Over-Estimates" = 2, "Prop GenAI Sig Under-Estimates" = 2)) 
writeLines(as.character(table_si5), "tables/table_si5.tex")

## Table SI6: Comparison of ratings by race of respondents ####
imgAvg <- pilotSubL %>% group_by(image, varLabel) %>%dplyr::summarize(mean = mean(value, na.rm = TRUE),
                                                                      sd = sd(value, na.rm = TRUE),
                                                                      median = median(value, na.rm = TRUE),
                                                                      nResp = n(),
                                                                      se = sd(value, na.rm = TRUE)/sqrt(n())) %>% 
  mutate(lower = mean - se,
         upper = mean + se,
         sample = "Prolific Representative") %>% distinct() 

# here we take everyone for whom race_3 ==1; so included mixed as long as black is part of that mix (instead of trying to capture 'black only' folks)
# create a binary variable based on race_3
pilotSubL$race_3_num<- as.numeric(pilotSubL$race_3)
pilotSubL$black<- ifelse(pilotSubL$race_3==1, 1, 0)
pilotSubL$black[is.na(pilotSubL$black)] <- 0

imgAvgRace <- pilotSubL %>% group_by(image, varLabel, black) %>%dplyr::summarize(mean = mean(value, na.rm = TRUE),
                                                                                 sd = sd(value, na.rm = TRUE),
                                                                                 median = median(value, na.rm = TRUE),
                                                                                 nResp = n(),
                                                                                 se = sd(value, na.rm = TRUE)/sqrt(n())) %>% 
  mutate(lower = mean - se,
         upper = mean + se,
         img = str_remove(image, "https://storage.googleapis.com/streetview_project_public/Detroit_processed/"),
         img = str_remove(img, ".jpg"),
         sample = "Prolific Representative") %>% distinct() 

# dmacs
imgAvgDMACS <- dmacsL %>%
  filter(!is.na(weights)) %>% 
  as_survey_design(weights = weights) %>% 
  group_by(img, varLabel) %>% 
  dplyr::summarize(mean = survey_mean(value, na.rm = TRUE, vartype = "se"),
                   sd = survey_sd(value, na.rm = TRUE),
                   median = survey_median(value, na.rm = TRUE),
                   nResp = n()) %>% 
  rename(se = mean_se) %>% 
  mutate(lower = mean - se,
         upper = mean + se,
         sample = "Detroit Sample") %>% distinct() 
imgAvg <- bind_rows(imgAvg, imgAvgDMACS) %>% filter(varLabel != "Average Income")

imgAvgRaceDMACS <- dmacsL %>% 
  filter(!is.na(weights) & !is.na(black)) %>% 
  as_survey_design(weights = weights) %>% 
  group_by(img, varLabel, black) %>%
  dplyr::summarize(mean = survey_mean(value, na.rm = TRUE, vartype = "se"),
                   sd = survey_sd(value, na.rm = TRUE),
                   median = survey_median(value, na.rm = TRUE),
                   nResp = n()) %>% 
  rename(se = mean_se) %>%
  mutate(lower = mean - se,
         upper = mean + se,
         sample = "Detroit Sample") %>% distinct() 

imgAvgRace <- bind_rows(imgAvgRace, imgAvgRaceDMACS)

# For each varLabel (Disorder, etc), we want to see if the means are significantly 
#different between black and non-black respondents, within each sample (Prolific vs Detroit):
race_compare_table2 <- imgAvgRace %>%
  mutate(attribute = case_when(
    varLabel == "Disorder" ~ "Disorder",
    varLabel == "Neighborhood Wealth" ~ "Wealth",
    varLabel == "Safety - Daytime Walking" ~ "Safety-Day",
    varLabel == "Safety - Nighttime Walking" ~ "Safety-Night",
    TRUE ~ varLabel
  )) %>%
  group_by(sample, black, attribute) %>%
  summarise(
    mean = mean(mean, na.rm = TRUE),
    median = mean(median, na.rm = TRUE),
    sd = mean(sd, na.rm = TRUE),
    nResp = mean(nResp, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  mutate(Group = paste0(ifelse(black == 1, "Black", "non-Black"), ", ",
                        ifelse(sample == "Prolific Representative", "U.S.", "Detroit"))) %>%
  select(attribute, Group, mean, median, sd, nResp) %>%
  pivot_wider(
    names_from = Group,
    values_from = c(mean, median, sd, nResp),
    names_glue = "{.value} {Group}"
  )

### table out ####
table_si6 <- kable(race_compare_table2, digits = 2,
                   caption = "Comparison of ratings by race of respondents",
                   format = "latex", row.names = FALSE, booktabs = TRUE, escape = FALSE, col.names = NULL) %>% 
  add_header_above(c(" " = 1, 
                     "non-Black" = 1, "Black" = 1, "non-Black" = 1, "Black" = 1, 
                     "non-Black" = 1, "Black" = 1, "non-Black" = 1, "Black" = 1,
                     "non-Black" = 1, "Black" = 1, "non-Black" = 1, "Black" = 1,
                     "non-Black" = 1, "Black" = 1, "non-Black" = 1, "Black" = 1)) %>% 
  add_header_above(c(" " = 1, 
                     "Detroit" = 2, "US" = 2, 
                     "Detroit" = 2, "US" = 2,
                     "Detroit" = 2, "US" = 2, 
                     "Detroit" = 2, "US" = 2)) %>% 
  add_header_above(c("Attribute" = 1, 
                     "Mean" = 4, 
                     "Median" = 4, 
                     "Std. Deviation" = 4,
                     "Num Respondents" = 4)) 
writeLines(as.character(table_si6), "tables/table_si6.tex")


## Table SI7: T-tests comparing mean ratings by race and sample ####

# we want to know, do Black Detroit residents provide ratings that are more similar 
# to non-Black Detroit residents than they are to Black U.S. residents?

# Clean attribute names first
imgAvgRace <- imgAvgRace %>%
  mutate(attribute = case_when(
    varLabel == "Disorder" ~ "Disorder",
    varLabel == "Neighborhood Wealth" ~ "Wealth",
    varLabel == "Safety - Daytime Walking" ~ "Safety-Day",
    varLabel == "Safety - Nighttime Walking" ~ "Safety-Night",
    TRUE ~ varLabel
  ))

# Function to compute two t-tests per attribute
run_comparisons <- function(attr_name) {
  # Subset data to this attribute
  dat <- imgAvgRace %>% filter(attribute == attr_name)
  
  # 1. Compare Black Detroit vs. Black U.S.
  black_comp <- t.test(
    mean ~ sample,
    data = dat %>% filter(black == 1)
  ) %>% tidy() %>%
    mutate(comparison = "Black Detroit vs Black U.S.", attribute = attr_name)
  
  # 2. Compare Black Detroit vs. non-Black Detroit
  det_comp <- t.test(
    mean ~ black,
    data = dat %>% filter(sample == "Detroit Sample")
  ) %>% tidy() %>%
    mutate(comparison = "Black vs non-Black Detroit", attribute = attr_name)
  
  bind_rows(black_comp, det_comp)
}

# Run across all attributes
all_ttests <- unique(imgAvgRace$attribute) %>%
  lapply(run_comparisons) %>%
  bind_rows()

# rename columns
all_ttests <- all_ttests %>%
  rename(Attribute = attribute, 
         Comparison = comparison,
         `Difference in Means` = estimate,
         `Mean 1` = estimate1,
         `Mean 2` = estimate2,
         `T-Stat` = statistic,
         `P-Value` = p.value) %>%
  select(Attribute, Comparison, 
         `Difference in Means`, 
         `Mean 1`, `Mean 2`, 
         `T-Stat`, `P-Value`)

### table out ####
table_si7<- xtable(all_ttests, caption = "T-tests comparing mean ratings by race and sample of respondents",
                   label = "tab:race_compare_ttests", digits = 2, include.rownames = FALSE)

print(table_si7, file = "tables/table_si7.tex")

## Table SI8: T-tests comparing mean ratings by race and sample ####
# compute two t-tests per attribute
run_comparisons <- function(attr_name) {
  # Subset data to this attribute
  dat <- imgAvgRace %>% filter(attribute == attr_name)
  
  # 1. Compare non-Black Detroit vs. non-Black U.S.
  nonblack_comp <- t.test(
    mean ~ sample,
    data = dat %>% filter(black == 0)
  ) %>% tidy() %>%
    mutate(comparison = "non-Black Detroit vs non-Black U.S.", attribute = attr_name)
  
  # 2. Compare Black US vs. non-Black US
  us_comp <- t.test(
    mean ~ black,
    data = dat %>% filter(sample == "Prolific Representative")
  ) %>% tidy() %>%
    mutate(comparison = "Black vs non-Black U.S.", attribute = attr_name)
  
  bind_rows(nonblack_comp, us_comp)
}

# Run across all attributes
all_ttests <- unique(imgAvgRace$attribute) %>%
  lapply(run_comparisons) %>%
  bind_rows()

# rename columns
all_ttests <- all_ttests %>%
  rename(Attribute = attribute, 
         Comparison = comparison,
         `Difference in Means` = estimate,
         `Mean 1` = estimate1,
         `Mean 2` = estimate2,
         `T-Stat` = statistic,
         `P-Value` = p.value) %>%
  select(Attribute, Comparison, 
         `Difference in Means`, 
         `Mean 1`, `Mean 2`, 
         `T-Stat`, `P-Value`)

### table out ####
table_si8<- xtable(all_ttests, caption = "T-tests comparing mean ratings by race and sample of respondents",
                   label = "tab:race_compare_ttests", digits = 2, include.rownames = FALSE)

print(table_si8, file = "tables/table_si8.tex")

## Table SI9: Correlations across race by sample ####
gptAvg <-gpt4o %>%
  group_by(`Image ID`, Question, intro) %>%
  summarise(nMissing = sum(is.na(Response)),
            meanGPT = mean(Response, na.rm=TRUE),
            seGPT = sd(Response, na.rm = TRUE) / sqrt(n()),
            sdGPT = sd(Response, na.rm = TRUE),
            nGPT = n())

imgAvgRaceDMACS <- dmacsL %>%
  mutate(black = ifelse(black == 1, "Black", "Non-Black")) %>%
  filter(!is.na(weights) & !is.na(black)) %>%
  as_survey_design(weights = weights) %>%
  group_by(img, varLabel, black) %>%
  dplyr::summarize(mean = survey_mean(value, na.rm = TRUE, vartype = "se"),
                   sd = survey_sd(value, na.rm = TRUE),
                   median = survey_median(value, na.rm = TRUE),
                   nResp = n()) %>%
  mutate(sd = replace_na(sd, 0)) %>% #causes NA when there is no variation
  rename(se = mean_se) %>%
  select(-median_se) %>%
  mutate(lower = mean - se,
         upper = mean + se,
         sample = "Detroit Sample",
         gptLabel = case_when(
           varLabel =="Safety - Nighttime Walking" ~ "Safety - night",
           varLabel =="Safety - Daytime Walking" ~ "Safety - day",
           varLabel == "Neighborhood Wealth" ~ "Wealth",
           .default = varLabel
         )) %>% distinct() %>%
  ungroup()

imgAvgRaceProl <- pilotSubL %>%
  mutate(black = ifelse(race == "Black or African American", "Black", "Non-Black")) %>%
  filter( !is.na(black)) %>%
  group_by(img, varLabel, black) %>%
  dplyr::summarize(mean = mean(value, na.rm = TRUE),
                   sd = sd(value, na.rm = TRUE),
                   median = median(value, na.rm = TRUE),
                   nResp = n(),
                   se = sd(value, na.rm = TRUE)/sqrt(n())) %>%
  mutate(sd = replace_na(sd, 0)) %>% #causes NA when there is no variation
  mutate(lower = mean - se,
         upper = mean + se,
         sample = "National Sample",
         gptLabel = case_when(
           varLabel =="Safety - Nighttime Walking" ~ "Safety - night",
           varLabel =="Safety - Daytime Walking" ~ "Safety - day",
           varLabel == "Neighborhood Wealth" ~ "Wealth",
           .default = varLabel
         )) %>% distinct() %>%
  ungroup()

raceImgsAvg <- bind_rows(imgAvgRaceProl, imgAvgRaceDMACS) %>%
  left_join(gptAvg %>% filter(intro == "No prompt"),
            by = c("img" = "Image ID", "gptLabel" = "Question")) %>%
  mutate(comparison = paste0(black, " ", sample, " v GPT"))

cor_dataRace <- raceImgsAvg %>% ungroup() %>%
  group_by(comparison, gptLabel) %>%
  summarise(
    correlation = cor(mean, meanGPT, use = "complete.obs"),
    n = sum(complete.cases(mean, meanGPT)),
    corrSE = (1 - correlation^2) / sqrt(n - 2)
  ) %>%
  ungroup() %>%
  arrange(gptLabel)

### table out ####
table_si9<- cor_dataRace %>% mutate_at(vars(correlation, n, corrSE), round, 2) %>%
  kable(format = "latex", booktabs = TRUE, row.names = FALSE, escape = FALSE) %>%
  kable_styling(latex_options = c("hold_position")) %>%
  add_header_above(c("Comparison" = 1, "Attribute" = 1, "Correlation" = 1, "N" = 1, "Correlation SE" = 1))

writeLines(as.character(table_si9), "tables/table_si9.tex")


## Table SI10: order ####
long_nt <- read_csv("Survey_data/pilot_data_order_analysis.csv")

#does the last five matter? #
orderFacReg <- lm(value ~ as_factor(image_order) + image + varLabel, 
                  long_nt)
orderReg <- lm(value ~ image_order + image + varLabel, 
               long_nt)

### table out ####
table_si10<- texreg::texreg(list(orderReg, orderFacReg),
                            custom.coef.map = list("(Intercept)" = "Intercept", "image_order" = "Image Order", "I(image_order^2)" = "(Image Order)^2",
                                                   "as_factor(image_order)2" = "Order: 2", 
                                                   "as_factor(image_order)3" = "Order: 3",
                                                   "as_factor(image_order)4" = "Order: 4",
                                                   "as_factor(image_order)5" = "Order: 5",
                                                   "as_factor(image_order)6" = "Order: 6",
                                                   "as_factor(image_order)7" = "Order: 7",
                                                   "as_factor(image_order)8" = "Order: 8",
                                                   "as_factor(image_order)9" = "Order: 9",
                                                   "as_factor(image_order)10" = "Order: 10"),
                            digits = 3, 
                            caption = "Effect of randomized image order on ratings of images. The regression image and question type fixed effects.",
                            label = "SI:imageOrder",
                            include.ci = FALSE,
)

writeLines(as.character(table_si10), "tables/table_si10.tex")


## Table SI11: demographics ####
dmacsSum <- dmacsL %>% distinct(sid, .keep_all = T) %>% select(gender_3cats, black, white, college, income_below_35k) %>% 
  mutate(women = ifelse(gender_3cats == 2, 1, 0) ) %>% 
  summarize_at(vars(black, white, college, income_below_35k, women), mean, na.rm = TRUE)

## add a row to summary table for weighed dmacs data 
dmacsSum_weighted<- dmacsL %>% distinct(sid, .keep_all = T) %>% select(gender_3cats, black, white, college, income_below_35k, weights) %>% 
  filter(!is.na(weights)) %>%
  mutate(women = ifelse(gender_3cats == 2, 1, 0) ) %>% 
  as_survey(weights = c(weights)) %>%
  summarize_at(vars(black, white, college, income_below_35k, women), funs(survey_mean(., na.rm=T)), na.rm = TRUE) 
#now just keep columns for black, white, college, income_below_35k, and women
dmacs_weighted <- dmacsSum_weighted %>% select(black, white, college, income_below_35k, women ) 

income_vals <- c('$10,000 to $14,999', '$5,000 to $9,999', 'Less than $5,000','$15,000 to $19,999',' $20,000 to $24,999', '$25,000 to $29,999','$30,000 to $34,999')
college_vals <- c("Bachelor's degree (for example, BA, BS, or AB)","Graduate degree (for example, Master's degree or doctorate)")
## combine the prolific, dmacs, and dmacs weighted tables to create summary tables 
prolificSum <- pilotSubL %>%distinct(ResponseId, .keep_all=T)%>% mutate(income_below_35k = ifelse(income %in% income_vals, 1, 0),
                                                                        women = ifelse(gender == "Female", 1, 0),
                                                                        college = ifelse(educ > 4, 1, 0),
                                                                        black = case_when(race_3 == 1 ~ 1, .default =0),
                                                                        white = case_when(race_5 == 1 ~ 1, .default =0)) %>% 
  summarize_at(vars(black, white, college, income_below_35k, women), mean, na.rm = TRUE)

sumTab <- prolificSum %>% mutate(sample = "U.S. Representative (Prolific)") %>% 
  bind_rows(dmacsSum %>%  mutate(sample = "Detroit Representative (DMACS)")) %>%
  bind_rows(dmacs_weighted %>% mutate(sample = "Detroit Representative (DMACS) - weighted"))

### table out ####
table_si11<- xtable::xtable(sumTab)
print(table_si11, file = "tables/table_si11.tex")

# Houses and cars per image ####
houses_cars <- read_csv("LMM_data/houses_cars_per_image.csv")

houses_cars %>% 
  group_by(question_text, `Image ID`) %>% 
  dplyr::summarize(meanResponse = mean(Response, na.rm = TRUE)) %>% 
  ungroup() %>% 
  group_by(question_text) %>% 
  dplyr::summarize(mean = mean(meanResponse, na.rm = TRUE), 
                   sd = sd(meanResponse, na.rm = TRUE),
                   numContain = sum(meanResponse>0, na.rm = TRUE))
