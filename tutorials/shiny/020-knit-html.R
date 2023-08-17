# This example shows you how to knit an R Markdown document using the two packages **knitr** and **markdown**, and include the HTML output in a shiny app. Note we are using [R Markdown v1](http://rmarkdown.rstudio.com/authoring_migrating_from_v1.html) here, which has a couple of differences with [v2](http://rmarkdown.rstudio.com). The basic idea is to call `knitr::knit2html()` with the argument `fragment.only = TRUE` to generate a fragment of HTML code.
# 

library(shiny)

# Workaround for https://github.com/yihui/knitr/issues/1538
if (packageVersion("knitr") < "1.17.3") {
  if (getRversion() > "3.4.0") {
    evaluate2 <- function(...) evaluate::evaluate(...)
    environment(evaluate2) <- asNamespace("knitr")
    knitr::knit_hooks$set(evaluate = evaluate2)
  } else {
    stop("Update knitr to run this app", call. = FALSE)
  }
}

server <- function(input, output) {

  regFormula <- reactive({
    as.formula(paste('mpg ~', input$x))
  })

  output$report <- renderUI({
    src <- normalizePath('report.Rmd')

    # temporarily switch to the temp dir, in case you do not have write
    # permission to the current working directory
    owd <- setwd(tempdir())
    on.exit(setwd(owd))
    knitr::opts_knit$set(root.dir = owd)

    tagList(
      HTML(knitr::knit2html(text = readLines(src), fragment.only = TRUE)),
      # typeset LaTeX math
      tags$script(HTML('MathJax.Hub.Queue(["Typeset", MathJax.Hub]);')),
      # syntax highlighting
      tags$script(HTML("if (hljs) $('#report pre code').each(function(i, e) {
                          hljs.highlightBlock(e)
                       });"))
    )
  })

}


ui <- fluidPage(
  title = 'Embed an HTML report from R Markdown/knitr',
  sidebarLayout(
    sidebarPanel(
      withMathJax(),  # include the MathJax library
      selectInput('x', 'Build a regression model of mpg against:',
                  choices = names(mtcars)[-1])
    ),
    mainPanel(
      uiOutput('report')
    )
  )
)

# Create Shiny app ----
shinyApp(ui = ui, server = server)
