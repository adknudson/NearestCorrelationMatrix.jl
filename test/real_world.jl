ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

push!(LOAD_PATH, @__DIR__)

using Test
import Datasets as DS # local module
import NearestCorrelationMatrix as NCM
using NearestCorrelationMatrix.Internals

@testset "BCCD16" begin
    A = DS.bccd16()
    @test iscorrelation(A) == false
    NCM.nearest_cor!(A)
    @test iscorrelation(A) == true
end

@testset "BEYU11" begin
    A = DS.beyu11()
    @test iscorrelation(A) == false
    NCM.nearest_cor!(A)
    @test iscorrelation(A) == true
end

@testset "BHWI01" begin
    A = DS.bhwi01()
    @test iscorrelation(A) == false
    NCM.nearest_cor!(A)
    @test iscorrelation(A) == true
end

@testset "COR1399" begin
    A = DS.cor1399()
    @test iscorrelation(A) == false
    NCM.nearest_cor!(A)
    @test iscorrelation(A) == true
end

@testset "COR3120" begin
    A = DS.cor3120()
    @test iscorrelation(A) == false
    NCM.nearest_cor!(A)
    @test iscorrelation(A) == true
end

# TODO: fing97. Need to implement masking

@testset "HIGH02" begin
    A = DS.high02()
    @test iscorrelation(A) == false
    NCM.nearest_cor!(A)
    @test iscorrelation(A) == true
end

@testset "MMB13" begin
    A = DS.mmb13()
    @test iscorrelation(A) == false
    NCM.nearest_cor!(A)
    @test iscorrelation(A) == true
end

@testset "TEC03" begin
    A = DS.tec03()
    @test iscorrelation(A) == false
    NCM.nearest_cor!(A)
    @test iscorrelation(A) == true
end

@testset "TYDA99R1" begin
    A = DS.tyda99r1()
    @test iscorrelation(A) == false
    NCM.nearest_cor!(A)
    @test iscorrelation(A) == true
end

@testset "TYDA99R2" begin
    A = DS.tyda99r2()
    @test iscorrelation(A) == false
    NCM.nearest_cor!(A)
    @test iscorrelation(A) == true
end

@testset "TYDA99R3" begin
    A = DS.tyda99r3()
    @test iscorrelation(A) == false
    NCM.nearest_cor!(A)
    @test iscorrelation(A) == true
end

# TODO: usgs13. Need to implement masking
