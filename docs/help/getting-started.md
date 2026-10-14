# Getting started

Create one complete case, run it, and inspect its results before building a
larger project or parameter study.

1. On the welcome screen, choose **Create new project**, give it a name, and
   select a local folder.
2. Choose **New Case** on the project screen. The case starts as a draft with
   numerical defaults; it still needs your physical inputs.
3. In **Chemistry**, select the gas and solid species and the required reactions.
4. In **Bed**, enter the geometry and complete every material zone.
5. In **Program**, set the initial feed and outlet pressure, then define timed
   steps or a repeating program with a suitable horizon.
6. In **General**, review the solver, sampling, and recorded quantities. Select
   the reports needed for the plots and Excel analysis you want.
7. Return to the project. When the case is **Ready**, choose its **Run** action.
8. After **Succeeded**, open the case and use **Show** beside an available plot
   in General. Use **Report** to prepare an Excel export.

Press **F1** in the application for a short explanation of the screen or case tab
you are using. The [GUI guide](../GUI-GUIDE.md) provides the detailed workflows.

## Keep comparisons

Each case keeps one latest run. To compare a changed case with a result you want
to retain, duplicate the case, edit the copy, and run that copy. **Results…** on
the project screen compares successful runs across cases.
